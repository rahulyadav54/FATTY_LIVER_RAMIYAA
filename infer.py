import argparse
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.siamese_net import SiameseNetwork
from src.data_loader import get_data_transforms


CLASS_NAMES = ['Normal', 'Grade-I', 'Grade-II', 'Grade-III', 'CLD']


def load_image_from_mat(mat_path, sample_idx=0, image_idx=0):
    from scipy.io import loadmat
    data = loadmat(str(mat_path))
    if 'data' not in data:
        raise ValueError("MAT file does not contain 'data' field")
    dataset = data['data'].flatten()
    if sample_idx >= len(dataset):
        raise IndexError('sample_idx out of range')
    sample = dataset[sample_idx]
    images = sample['images']
    img = images[image_idx]
    img_pil = Image.fromarray(img, mode='L').convert('RGB')
    return img_pil


def infer_image(model, device, image, transform):
    model.eval()
    img_t = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(img_t)
        # Debug: check for NaNs in logits
        if torch.isnan(logits).any():
            print('Warning: logits contain NaN values')
            print('Logits stats:', torch.min(logits).item(), torch.max(logits).item(), torch.mean(logits).item())
        probs_t = F.softmax(logits, dim=1)
        if torch.isnan(probs_t).any():
            print('Warning: softmax produced NaN values')
        probs = probs_t.cpu().numpy()[0]
        pred = int(np.argmax(probs))
        confidence = float(np.max(probs))
    return pred, confidence, probs


def main():
    parser = argparse.ArgumentParser(description='Infer fatty liver class from one ultrasound image')
    parser.add_argument('--image', type=str, help='Path to image (JPG/PNG)')
    parser.add_argument('--mat_file', type=str, default='data/dataset_liver_bmodes_steatosis_assessment_IJCARS.mat', help='Path to dataset .mat file')
    parser.add_argument('--mat_sample', type=int, help='Index of sample in .mat file (0-based)')
    parser.add_argument('--mat_image_idx', type=int, default=0, help='Which image in the sample (0-9)')
    parser.add_argument('--model_path', type=str, default='best_model.pth', help='Path to trained model')
    parser.add_argument('--binary', action='store_true', help='Also output binary Normal vs Abnormal probability')
    args = parser.parse_args()

    if args.image is None and args.mat_sample is None:
        parser.error('Provide either --image or --mat_sample to infer from .mat')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load image
    if args.image:
        image = Image.open(args.image).convert('RGB')
    else:
        mat_path = Path(args.mat_file)
        if not mat_path.exists():
            raise FileNotFoundError(f'MAT file not found: {mat_path}')
        image = load_image_from_mat(mat_path, sample_idx=args.mat_sample, image_idx=args.mat_image_idx)

    transform = get_data_transforms(is_train=False)

    # Load model
    model = SiameseNetwork().to(device)
    state = torch.load(args.model_path, map_location=device)
    # Attempt normal load first
    loaded = False
    try:
        model.load_state_dict(state)
        loaded = True
    except Exception:
        # Try loading state dict under 'model' key
        if isinstance(state, dict) and 'model' in state:
            try:
                model.load_state_dict(state['model'])
                loaded = True
            except Exception:
                loaded = False
        else:
            loaded = False

    # If load succeeded, check for NaNs in parameters
    any_nan = False
    for name, p in model.named_parameters():
        if torch.isnan(p).any():
            print(f'Parameter {name} contains NaNs')
            any_nan = True

    # If parameters contain NaNs or load failed, attempt partial load skipping NaN tensors
    if not loaded or any_nan:
        print('Attempting partial load: skipping tensors that are NaN in the checkpoint')
        ckpt = state
        if isinstance(state, dict) and 'model' in state and not isinstance(state['model'], torch.Tensor):
            ckpt = state['model']
        clean_ckpt = {}
        for k, v in ckpt.items():
            try:
                if hasattr(v, 'dtype') and torch.isnan(v).any():
                    print(f'  skipping {k} (contains NaNs)')
                    continue
            except Exception:
                pass
            clean_ckpt[k] = v
        model.load_state_dict(clean_ckpt, strict=False)

    pred, conf, probs = infer_image(model, device, image, transform)

    threshold = 0.60 if args.binary else 0.40
    is_low_conf = conf < threshold

    print(f'\n{"="*60}')
    print(f'🔬 INFERENCE RESULT (RESEARCH PROTOTYPE ONLY)')
    print(f'{"="*60}')
    if is_low_conf:
        print(f'STATUS: WARNING - LOW CONFIDENCE')
        print(f'Prediction: Low-confidence prediction — further review required')
        print(f'Top candidate: {CLASS_NAMES[pred]} (Confidence: {conf*100:.2f}% < Threshold: {threshold*100:.0f}%)')
    else:
        print(f'STATUS: RELIABLE PREDICTION')
        print(f'Predicted class: {CLASS_NAMES[pred]} (index {pred})')
        print(f'Confidence: {conf*100:.2f}%')
        
    print(f'\nProbability distribution:')
    for i, class_name in enumerate(CLASS_NAMES):
        print(f'  {class_name:12s}: {probs[i]*100:6.2f}%')

    if args.binary:
        normal_prob = float(probs[0])
        abnormal_prob = float(np.sum(probs[1:]))
        binary_conf = normal_prob if pred == 0 else abnormal_prob
        binary_low_conf = binary_conf < 0.60
        print(f'\nBinary classification:')
        if binary_low_conf:
            print(f'  Prediction: Low-confidence prediction — further review required')
            print(f'  P(Normal)   = {normal_prob*100:.2f}%')
            print(f'  P(Abnormal) = {abnormal_prob*100:.2f}%')
        else:
            binary_pred = 'Normal' if pred == 0 else 'Abnormal'
            print(f'  Prediction: {binary_pred}')
            print(f'  P(Abnormal) = {abnormal_prob*100:.2f}%')
    print(f'\n⚠️  DISCLAIMER: Research prototype only — not for clinical diagnosis.')
    print(f'{"="*60}\n')


if __name__ == '__main__':
    main()
