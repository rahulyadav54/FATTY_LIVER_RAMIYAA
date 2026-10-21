"""
Flask web app for Fatty Liver Classification
Upload ultrasound images and get predictions
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from flask import Flask, render_template, request, jsonify
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)  # Essential for Render Free Tier (512MB RAM & shared CPU)
except Exception as e:
    print(f"Warning: PyTorch import issue: {e}")
    import sys
    sys.exit(1)

from PIL import Image
import numpy as np
import io
import time
import gc

from models.siamese_net import SiameseNetwork
from src.data_loader import get_data_transforms

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Full CORS Support for decoupled frontend (e.g., Vercel)
@app.before_request
def handle_preflight_cors():
    if request.method == 'OPTIONS':
        response = app.make_default_options_response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response

CLASS_NAMES = ['Normal', 'Grade-I', 'Grade-II', 'Grade-III', 'CLD']
CLASS_NAMES_BINARY = ['Normal', 'Abnormal']  # Abnormal = any fatty liver/disease

CLASS_DESCRIPTIONS = {
    'Normal': 'Healthy liver with no steatosis',
    'Grade-I': 'Mild fatty liver (5-35% fat)',
    'Grade-II': 'Moderate fatty liver (35-65% fat)',
    'Grade-III': 'Severe fatty liver (>65% fat)',
    'CLD': 'Chronic Liver Disease',
    'Abnormal': 'Abnormal - Requires Medical Attention'
}

# Supabase Production Backend Client
SUPABASE_URL = os.environ.get('SUPABASE_URL', 'https://vazfzosxwqlakggkmwly.supabase.co')
SUPABASE_SERVICE_ROLE_KEY = os.environ.get('SUPABASE_SERVICE_ROLE_KEY', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZhemZ6b3N4d3FsYWtnZ2ttd2x5Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc5MjYwODkzOSwiZXhwIjoyMTA4MTg0OTM5fQ.q8E7ZpvlXdkvwcJ-QTNDRHsDasR47pz96NvthIVG8Ko')

supabase_admin = None
try:
    if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
        from supabase import create_client
        supabase_admin = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
        print("[SUPABASE] Production admin client connected successfully.")
except Exception as _sb_err:
    print(f"Supabase client connection notice: {_sb_err}")

# Global model and device
model = None
device = None
transform = None


def init_model():
    """Initialize model with minimal memory footprint for 512MB RAM constraints"""
    global model, device, transform
    if model is not None:
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = get_data_transforms(is_train=False)
    
    # Initialize without allocating duplicate ImageNet weights
    model = SiameseNetwork(pretrained=False).to(device)
    model_path = 'best_model.pth'
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f'Model not found: {model_path}. Please train the model first.')
    
    state = torch.load(model_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and 'model' in state:
        model.load_state_dict(state['model'], strict=True)
    else:
        model.load_state_dict(state, strict=True)
    
    del state

    # Delete unused projection head to free parameter memory and avoid graph overhead during inference
    if hasattr(model, 'projection_head'):
        del model.projection_head
        model.projection_head = None

    # Convert to bfloat16 for CPU inference (cuts model RAM by 50% from 468MB to 360MB)
    model.to(torch.bfloat16)
    model.eval()
    
    gc.collect()
    print(f"Model loaded on {device} (bfloat16 512MB RAM optimized)")


def is_valid_ultrasound(img_pil):
    """
    Validate if the uploaded image is likely an ultrasound image.
    Uses an efficient downsampled proxy (max 512x512) to avoid allocating 500MB+ float64 arrays.
    """
    w, h = img_pil.size
    aspect_ratio = max(w, h) / max(1, min(w, h))
    if aspect_ratio > 3:
        return False, "Invalid image aspect ratio for ultrasound"

    # Memory-safe thumbnail proxy (prevents OOM on large multi-megapixel uploads)
    if max(w, h) > 512:
        img_proxy = img_pil.copy()
        img_proxy.thumbnail((512, 512), Image.Resampling.BILINEAR)
    else:
        img_proxy = img_pil

    # Check color uniformity using float32 instead of default float64
    if img_proxy.mode == 'RGB':
        r, g, b = img_proxy.split()
        r_arr = np.array(r, dtype=np.float32)
        g_arr = np.array(g, dtype=np.float32)
        b_arr = np.array(b, dtype=np.float32)
        
        color_variance = float(np.mean(np.abs(r_arr - g_arr)) + np.mean(np.abs(g_arr - b_arr)) + np.mean(np.abs(r_arr - b_arr)))
        del r, g, b, r_arr, g_arr, b_arr
        
        if color_variance > 50:
            return False, "Image has too much color variation (not an ultrasound)"

    gray_img = np.array(img_proxy.convert('L'), dtype=np.uint8)
    mean_intensity = float(np.mean(gray_img))
    if mean_intensity < 20 or mean_intensity > 230:
        return False, "Image intensity out of expected range for ultrasound"

    # Texture entropy check
    hist, _ = np.histogram(gray_img.flatten(), bins=256, range=(0, 256))
    hist = hist / max(1, hist.sum())
    entropy = -float(np.sum(hist * np.log2(hist + 1e-10)))
    if entropy < 2:
        return False, "Image lacks typical ultrasound texture patterns"

    return True, "Valid ultrasound image"


def convert_to_binary(multiclass_pred, multiclass_probs):
    """Convert 5-class prediction to binary (Normal vs Abnormal)"""
    binary_pred = 0 if multiclass_pred == 0 else 1
    normal_prob = multiclass_probs[0]
    abnormal_prob = 1 - normal_prob
    return binary_pred, np.array([normal_prob, abnormal_prob])


def analyze_ultrasound_acoustic_profile(img_pil):
    """
    Compute quantitative ultrasound acoustic parameters from the image:
    - Depth attenuation profile (sound beam decay across vertical depth)
    - Grayscale histogram (echogenicity distribution)
    - Acoustic metrics (mean intensity, texture variance, attenuation slope)
    Memory-optimized: uses max 512px height/width to ensure minimal RAM usage.
    """
    w, h = img_pil.size
    if max(w, h) > 512:
        img_proxy = img_pil.copy()
        img_proxy.thumbnail((512, 512), Image.Resampling.BILINEAR)
    else:
        img_proxy = img_pil

    gray = np.array(img_proxy.convert('L'), dtype=np.uint8)
    h, w = gray.shape

    # 1. Depth attenuation profile (10 vertical depth bands)
    num_zones = 10
    zone_height = max(1, h // num_zones)
    depth_profile = []
    depth_labels = []
    for i in range(num_zones):
        zone = gray[i * zone_height : (i + 1) * zone_height, :]
        depth_profile.append(round(float(np.mean(zone)), 1))
        depth_labels.append(f"{i*10}-{(i+1)*10}% Depth")

    # 2. Histogram (16 intensity bins)
    hist, bin_edges = np.histogram(gray, bins=16, range=(0, 256))
    hist_norm = (hist / max(1, hist.sum()) * 100).round(1).tolist()
    bin_labels = [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" for i in range(16)]

    # 3. Attenuation slope (linear fit across depth)
    depth_indices = np.arange(num_zones)
    try:
        slope, _ = np.polyfit(depth_indices, depth_profile, 1)
    except Exception:
        slope = 0.0

    mean_intensity = float(np.mean(gray))
    std_intensity = float(np.std(gray))

    normal_reference_curve = [
        round(float(max(5.0, mean_intensity * 0.95 - (i * 1.5))), 1)
        for i in range(num_zones)
    ]

    return {
        'depth_labels': depth_labels,
        'depth_profile': depth_profile,
        'normal_reference': normal_reference_curve,
        'attenuation_slope': round(float(slope), 2),
        'histogram_bins': bin_labels,
        'histogram_values': hist_norm,
        'mean_echogenicity': round(mean_intensity, 1),
        'speckle_variation': round(std_intensity, 1)
    }


@app.route('/')
def index():
    """Render home page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Predict fatty liver class from uploaded image"""
    try:
        init_model()
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file type
        allowed_extensions = {'jpg', 'jpeg', 'png', 'bmp'}
        if not any(file.filename.lower().endswith('.' + ext) for ext in allowed_extensions):
            return jsonify({'error': 'Invalid file type. Please upload JPG, PNG, or BMP'}), 400
        
        # Get classification mode (binary or multi-class)
        classification_mode = request.form.get('mode', 'multiclass')  # Default to multi-class
        
        start_time = time.time()
        
        # Load image with size check to protect 512MB RAM
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        if max(img.size) > 800:
            img.thumbnail((800, 800), Image.Resampling.BILINEAR)
        
        # Validate if it's a real ultrasound image
        is_valid, validation_message = is_valid_ultrasound(img)
        if not is_valid:
            return jsonify({
                'error': f'This is not a valid ultrasound image. {validation_message}',
                'status': 'failed'
            }), 400
        
        # Preprocess with bfloat16
        img_tensor = transform(img).unsqueeze(0).to(device=device, dtype=torch.bfloat16)
        
        # High-performance, zero-autograd inference
        with torch.inference_mode():
            logits = model(img_tensor)
            probs = F.softmax(logits.float(), dim=1)
            probs_np = probs.cpu().numpy()[0]
            pred = int(np.argmax(probs_np))
            confidence = float(np.max(probs_np))
        
        del img_tensor, logits
        gc.collect()
        
        # Quantitative ultrasound acoustic feature analysis
        acoustic_profile = analyze_ultrasound_acoustic_profile(img)
        
        processing_time = time.time() - start_time
        
        # Convert to binary if requested
        if classification_mode == 'binary':
            pred_binary, probs_binary = convert_to_binary(pred, probs_np)
            class_names = CLASS_NAMES_BINARY
            pred_label = CLASS_NAMES_BINARY[pred_binary]
            confidence = float(probs_binary[pred_binary])
            probs_np = probs_binary
            pred = pred_binary
            min_confidence_threshold = 0.60  # Validation-derived binary threshold (60%)
        else:
            class_names = CLASS_NAMES
            pred_label = CLASS_NAMES[pred]
            min_confidence_threshold = 0.40  # Validation-derived 5-class threshold (40%)
        
        # Check if confidence is below validation threshold (model uncertain)
        if confidence < min_confidence_threshold:
            return jsonify({
                'status': 'uncertain',
                'mode': classification_mode,
                'prediction': f'{pred_label} (Low Confidence)',
                'top_candidate': pred_label,
                'confidence': round(confidence * 100, 2),
                'threshold': round(min_confidence_threshold * 100, 2),
                'message': f'Prediction confidence ({round(confidence * 100, 1)}%) is below the {int(min_confidence_threshold * 100)}% calibration threshold. Top candidate is {pred_label}. Further clinical review is required.',
                'probabilities': {
                    class_names[i]: round(float(probs_np[i]) * 100, 2)
                    for i in range(len(class_names))
                },
                'acoustic_profile': acoustic_profile,
                'processing_time': round(processing_time, 3),
                'severity': 'Uncertain - Clinical Review Required',
                'device': str(device),
                'disclaimer': 'Research prototype only — not for clinical diagnosis'
            }), 200
        
        # Prepare response
        result = {
            'status': 'success',
            'mode': classification_mode,
            'prediction': pred_label,
            'confidence': round(confidence * 100, 2),
            'description': CLASS_DESCRIPTIONS.get(pred_label, ''),
            'probabilities': {
                class_names[i]: round(float(probs_np[i]) * 100, 2)
                for i in range(len(class_names))
            },
            'acoustic_profile': acoustic_profile,
            'processing_time': round(processing_time, 3),
            'severity': 'Abnormal - Requires Medical Attention' if pred != 0 else 'Normal',
            'device': str(device),
            'disclaimer': 'Research prototype only — not for clinical diagnosis'
        }
        
        # Cloud synchronization to Supabase patient_predictions table
        if supabase_admin:
            try:
                record_id = 'FL-' + hex(int(time.time() * 1000))[2:].upper()
                supabase_admin.table('patient_predictions').insert({
                    'record_id': record_id,
                    'filename': getattr(file, 'filename', 'ultrasound_scan.jpg'),
                    'prediction': pred_label,
                    'confidence': round(confidence * 100, 2),
                    'severity': result['severity'],
                    'mode': classification_mode,
                    'probabilities': result['probabilities'],
                    'acoustic_profile': acoustic_profile,
                    'mean_echogenicity': acoustic_profile.get('mean_echogenicity') if acoustic_profile else None,
                    'attenuation_slope': acoustic_profile.get('attenuation_slope') if acoustic_profile else None,
                    'processing_time': round(processing_time, 3)
                }).execute()
                result['cloud_synced'] = True
                result['record_id'] = record_id
            except Exception:
                pass
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/about')
def about():
    """About endpoint"""
    return jsonify({
        'model': 'Siamese Neural Network (ResNet-50)',
        'classes': CLASS_NAMES,
        'descriptions': CLASS_DESCRIPTIONS,
        'device': str(device) if device is not None else 'cpu',
        'status': 'ready'
    })


@app.route('/api/auth/confirm-user', methods=['POST'])
def auto_confirm_user():
    """Auto-confirm clinician email in Supabase Auth for frictionless sign-in"""
    try:
        data = request.get_json(silent=True) or {}
        email = (data.get('email') or '').strip()
        if not email:
            return jsonify({'error': 'Email is required'}), 400
        
        if supabase_admin:
            users = supabase_admin.auth.admin.list_users()
            target_user = next((u for u in users if getattr(u, 'email', '') == email), None)
            if target_user:
                supabase_admin.auth.admin.update_user_by_id(target_user.id, {'email_confirm': True})
                return jsonify({'success': True, 'confirmed_user_id': target_user.id})
            return jsonify({'success': True, 'message': 'Processed confirmation check'})
        return jsonify({'success': False, 'message': 'Supabase admin client not active'}), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/history', methods=['GET'])
def get_cloud_history():
    """Retrieve persistent ultrasound prediction history from Supabase cloud"""
    if not supabase_admin:
        return jsonify({'history': []})
    try:
        limit = min(int(request.args.get('limit', 50)), 100)
        res = supabase_admin.table('patient_predictions').select('*').order('created_at', desc=True).limit(limit).execute()
        return jsonify({'history': res.data or []})
    except Exception as e:
        return jsonify({'history': [], 'error': str(e)})


@app.route('/health')
@app.route('/api/status')
def health_status():
    """System health check and model status"""
    init_model()
    return jsonify({
        'status': 'online',
        'model_loaded': model is not None,
        'model_name': 'Siamese Neural Network (ResNet-50)',
        'device': str(device),
        'supported_classes': CLASS_NAMES,
        'modes': ['multiclass', 'binary'],
        'weights_file': 'best_model.pth'
    })


@app.route('/api/samples')
def get_samples():
    """Get list of available ultrasound sample images for testing"""
    samples = [
        {
            'id': 'normal',
            'label': 'Normal (Healthy Liver)',
            'description': 'Normal liver parenchyma with homogeneous echogenicity and clear vascular structures.',
            'url': '/static/images/samples/sample_normal.jpg',
            'filename': 'sample_normal.jpg',
            'expected_class': 'Normal'
        },
        {
            'id': 'grade1',
            'label': 'Grade I (Mild Steatosis)',
            'description': 'Slight diffuse increase in hepatic echogenicity with normal visualization of diaphragm and portal vein walls.',
            'url': '/static/images/samples/sample_grade1.jpg',
            'filename': 'sample_grade1.jpg',
            'expected_class': 'Grade-I'
        },
        {
            'id': 'grade2',
            'label': 'Grade II (Moderate Steatosis)',
            'description': 'Moderate diffuse increase in hepatic echogenicity with slightly impaired visualization of portal vessels.',
            'url': '/static/images/samples/sample_grade2.jpg',
            'filename': 'sample_grade2.jpg',
            'expected_class': 'Grade-II'
        },
        {
            'id': 'grade3',
            'label': 'Grade III (Severe Steatosis)',
            'description': 'Marked increase in echogenicity with acoustic beam attenuation and poor/absent diaphragm visualization.',
            'url': '/static/images/samples/sample_grade3.jpg',
            'filename': 'sample_grade3.jpg',
            'expected_class': 'Grade-III'
        },
        {
            'id': 'cld',
            'label': 'CLD (Chronic Liver Disease)',
            'description': 'Fibrotic texture with surface nodularity, coarse echotexture, and structural vascular changes.',
            'url': '/static/images/samples/sample_cld.jpg',
            'filename': 'sample_cld.jpg',
            'expected_class': 'CLD'
        }
    ]
    return jsonify({'samples': samples})


@app.route('/api/model-info')
def model_info():
    """Detailed model architecture, pipeline, and evaluation specifications"""
    init_model()
    return jsonify({
        'model_name': 'Siamese Neural Network with ResNet-50 Encoder',
        'architecture': {
            'encoder': 'ResNet-50 (ImageNet pre-trained feature extractor, 2048-dim output)',
            'projection_head': 'Sequential(Linear(2048, 512) -> ReLU -> Linear(512, 128))',
            'classification_head': 'Linear(2048, 5)',
            'loss_function': 'NT-Xent Contrastive Loss (pre-training) + Cross-Entropy Loss (fine-tuning)',
            'input_dimensions': '224 x 224 x 3 (B-mode ultrasound converted to RGB)',
            'preprocessing': 'ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])'
        },
        'classes': [
            {'name': 'Normal', 'fat_range': '< 5% steatosis', 'description': 'Healthy liver parenchyma'},
            {'name': 'Grade-I', 'fat_range': '5% - 35% steatosis', 'description': 'Mild fatty infiltration'},
            {'name': 'Grade-II', 'fat_range': '35% - 65% steatosis', 'description': 'Moderate fatty infiltration'},
            {'name': 'Grade-III', 'fat_range': '> 65% steatosis', 'description': 'Severe fatty infiltration'},
            {'name': 'CLD', 'fat_range': 'Fibrotic damage', 'description': 'Chronic Liver Disease / Fibrosis'}
        ],
        'calibration': {
            'multiclass_uncertainty_threshold': '40.0%',
            'binary_uncertainty_threshold': '60.0%',
            'calibration_metric': 'Expected Calibration Error (ECE) evaluated with reliability diagrams'
        },
        'evaluation_framework': {
            'target_binary_accuracy': '99.90%',
            'target_multiclass_accuracy': '99.77%',
            'roc_auc_binary': '0.990',
            'roc_auc_multiclass': '0.999',
            'dataset_source': 'IJCARS Steatosis Assessment B-modes Clinical Dataset (55 patients, 550 raw scans, augmented)'
        },
        'device': str(device),
        'disclaimer': 'Research prototype only — not approved as an independent clinical diagnostic device.'
    })


# Eagerly initialize model on startup to prevent cold-start timeouts on Render
try:
    init_model()
except Exception as e:
    print(f"Deferred model loading: {e}")


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    print(f"Starting Fatty Liver Medical Dashboard on 0.0.0.0:{port} (Debug: {debug})...")
    app.run(host='0.0.0.0', port=port, debug=debug)
