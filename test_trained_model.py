"""
Test script to validate trained model performance
Compares predictions before and after training
"""
import sys
import os
from pathlib import Path
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.data_loader import get_data_loaders
from models.siamese_net import SiameseNetwork
import numpy as np
from tqdm import tqdm

def test_model(model, test_loader, device):
    """Test model performance"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate accuracy
    accuracy = (all_preds == all_labels).mean()
    
    # Calculate per-class metrics
    class_names = ['Normal', 'Grade-I', 'Grade-II', 'Grade-III', 'CLD']
    
    return accuracy, all_preds, all_labels, all_probs, class_names

def main():
    print("=" * 70)
    print("🧪 MODEL VALIDATION TEST")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✅ Device: {device}\n")
    
    # Load data
    print("📁 Loading test dataset...")
    train_loader, val_loader, test_loader = get_data_loaders(
        data_dir='data',
        batch_size=16,
        num_workers=0
    )
    print(f"✅ Test samples: {len(test_loader.dataset)}\n")
    
    # Load trained model
    print("🧠 Loading trained model...")
    model = SiameseNetwork().to(device)
    model_path = 'best_model.pth'
    
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and 'model' in state:
        model.load_state_dict(state['model'], strict=True)
    else:
        model.load_state_dict(state, strict=True)
    
    print(f"✅ Model loaded from {model_path}\n")
    
    # Test model
    print("🔍 Running predictions on test set...\n")
    accuracy, preds, labels, probs, class_names = test_model(model, test_loader, device)
    
    # Print results
    print("\n" + "=" * 70)
    print(f"📊 OVERALL ACCURACY: {accuracy:.2%}")
    print("=" * 70)
    
    # Per-class accuracy
    print("\n📈 PER-CLASS PERFORMANCE:")
    print("-" * 70)
    for class_idx, class_name in enumerate(class_names):
        mask = labels == class_idx
        if mask.sum() > 0:
            class_acc = (preds[mask] == labels[mask]).mean()
            class_count = mask.sum()
            print(f"   {class_name:15} | Accuracy: {class_acc:6.2%} | Samples: {class_count:3}")
    
    # Confidence analysis
    max_probs = probs.max(axis=1)
    avg_confidence = max_probs.mean()
    min_confidence = max_probs.min()
    max_confidence = max_probs.max()
    
    print("\n🎯 CONFIDENCE ANALYSIS:")
    print("-" * 70)
    print(f"   Average Confidence: {avg_confidence:.2%}")
    print(f"   Min Confidence:     {min_confidence:.2%}")
    print(f"   Max Confidence:     {max_confidence:.2%}")
    
    # Show sample predictions
    print("\n📋 SAMPLE PREDICTIONS (First 10):")
    print("-" * 70)
    for i in range(min(10, len(preds))):
        pred_label = class_names[preds[i]]
        true_label = class_names[labels[i]]
        confidence = max_probs[i]
        match = "✅" if preds[i] == labels[i] else "❌"
        print(f"   {match} Pred: {pred_label:12} | True: {true_label:12} | Conf: {confidence:.2%}")
    
    print("\n" + "=" * 70)
    if accuracy >= 0.80:
        print("✅ MODEL PERFORMANCE: EXCELLENT")
        print("   Ready for deployment! High accuracy achieved.")
    elif accuracy >= 0.70:
        print("⚠️  MODEL PERFORMANCE: GOOD")
        print("   Acceptable for use, but could benefit from more training.")
    else:
        print("❌ MODEL PERFORMANCE: POOR")
        print("   Model needs more training or tuning.")
    print("=" * 70)

if __name__ == '__main__':
    main()
