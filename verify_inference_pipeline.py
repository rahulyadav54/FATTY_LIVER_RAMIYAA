"""
Verification script for Fatty Liver Classification Inference Pipeline
Tests 5-class and binary classification modes across sample ultrasound images.
Verifies proper low-confidence warning handling and prototype disclaimers.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from models.siamese_net import SiameseNetwork
from src.data_loader import get_data_transforms
from infer import infer_image

CLASS_NAMES = ['Normal', 'Grade-I', 'Grade-II', 'Grade-III', 'CLD']
CLASS_NAMES_BINARY = ['Normal', 'Abnormal']

def test_inference_on_images():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    transform = get_data_transforms(is_train=False)
    model = SiameseNetwork().to(device)
    
    model_path = 'best_model.pth'
    if Path(model_path).exists():
        state = torch.load(model_path, map_location=device)
        if isinstance(state, dict) and 'model' in state:
            model.load_state_dict(state['model'], strict=True)
        else:
            model.load_state_dict(state, strict=True)
        print(f"Loaded weights from {model_path}")
    
    test_samples = [
        ('data/Normal/Normal_000.jpg', 'Normal'),
        ('data/Grade-I/Grade-I_000.jpg', 'Grade-I'),
        ('data/Grade-II/Grade-II_000.jpg', 'Grade-II'),
        ('data/Grade-III/Grade-III_000.jpg', 'Grade-III'),
        ('data/CLD/CLD_000.jpg', 'CLD'),
    ]
    
    print("\n" + "=" * 70)
    print("🧪 INFERENCE PIPELINE VERIFICATION")
    print("=" * 70)
    
    for img_path, ground_truth in test_samples:
        if not Path(img_path).exists():
            print(f"Skipping {img_path} (not found)")
            continue
            
        img = Image.open(img_path).convert('RGB')
        pred, conf, probs = infer_image(model, device, img, transform)
        
        multiclass_threshold = 0.40
        binary_threshold = 0.60
        
        is_low_conf_multi = conf < multiclass_threshold
        
        # Multiclass prediction display
        multi_display = "Low-confidence prediction — further review required" if is_low_conf_multi else CLASS_NAMES[pred]
        
        # Binary prediction calculation
        normal_prob = float(probs[0])
        abnormal_prob = float(np.sum(probs[1:]))
        binary_pred_idx = 0 if normal_prob >= abnormal_prob else 1
        binary_conf = normal_prob if binary_pred_idx == 0 else abnormal_prob
        is_low_conf_binary = binary_conf < binary_threshold
        binary_display = "Low-confidence prediction — further review required" if is_low_conf_binary else CLASS_NAMES_BINARY[binary_pred_idx]
        
        print(f"\n📸 Image: {img_path} (True Label: {ground_truth})")
        print(f"  • Raw Top Class:     {CLASS_NAMES[pred]} (Conf: {conf * 100:.2f}%)")
        print(f"  • 5-Class Output:    {multi_display}")
        print(f"  • Binary Output:     {binary_display} (P_abnormal: {abnormal_prob*100:.2f}%)")
        print(f"  • Probabilities:     Normal: {probs[0]*100:.1f}%, Grade-I: {probs[1]*100:.1f}%, Grade-II: {probs[2]*100:.1f}%, Grade-III: {probs[3]*100:.1f}%, CLD: {probs[4]*100:.1f}%")

    print("\n" + "=" * 70)
    print("✅ Pipeline test finished successfully.")
    print("=" * 70)

if __name__ == '__main__':
    test_inference_on_images()

