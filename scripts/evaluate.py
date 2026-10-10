import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import sys
import os
from pathlib import Path
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from models.siamese_net import SiameseNetwork
from src.data_loader import get_data_loaders

def evaluate_model(model, data_loader, device, num_classes=5):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in data_loader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)

def binary_classification_metrics(preds, labels):
    # Binary: Normal (0) vs Abnormal (1-4)
    binary_preds = (preds > 0).astype(int)
    binary_labels = (labels > 0).astype(int)
    
    accuracy = np.mean(binary_preds == binary_labels)
    return accuracy

def main():
    parser = argparse.ArgumentParser(description='Evaluate Siamese Network for Fatty Liver Classification')
    parser.add_argument('--data_dir', type=str, default='data', help='Path to dataset')
    parser.add_argument('--model_path', type=str, default='best_model.pth', help='Path to trained model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    try:
        _, _, test_loader = get_data_loaders(args.data_dir, args.batch_size)
    except Exception as e:
        print(f"Error loading data: {e}")
        print(f"Make sure dataset exists at: {args.data_dir}")
        sys.exit(1)
    
    # Load model
    try:
        model = SiameseNetwork().to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Model loaded from: {args.model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Make sure model exists at: {args.model_path}")
        sys.exit(1)
    
    # Evaluate
    preds, labels, probs = evaluate_model(model, test_loader, device)
    
    # Multi-class metrics
    print("Multi-class Classification Report:")
    unique_labels = np.unique(labels)
    print(f"Unique labels found: {unique_labels}")

    # Map present label indices to readable class names
    class_names = ["Normal", "Grade-I", "Grade-II", "Grade-III", "CLD"]
    target_names = [class_names[int(lbl)] if int(lbl) < len(class_names) else f"Class-{int(lbl)}" for lbl in unique_labels]
    print(classification_report(labels, preds, labels=unique_labels, target_names=target_names))
    
    print("Confusion Matrix:")
    print(confusion_matrix(labels, preds))
    
    # Binary classification accuracy
    binary_acc = binary_classification_metrics(preds, labels)
    print(f"Binary Classification Accuracy (Normal vs Abnormal): {binary_acc:.4f}")
    
    # ROC-AUC for multi-class
    try:
        labels_int = labels.astype(int)
        probs_arr = np.array(probs)

        # Basic validation
        if probs_arr.ndim != 2:
            raise ValueError(f"probs must be a 2D array of shape (n_samples, n_classes), got {probs_arr.shape}")
        if probs_arr.shape[0] != labels_int.shape[0]:
            raise ValueError(f"mismatch: number of prob rows {probs_arr.shape[0]} != number of labels {labels_int.shape[0]}")

        unique_classes = np.unique(labels_int)
        if len(unique_classes) > 1:
            # Only use columns for classes that actually appear in the test set
            probs_filtered = probs_arr[:, unique_classes]
            probs_norm = probs_filtered / np.sum(probs_filtered, axis=1, keepdims=True)
            roc_auc = roc_auc_score(labels_int, probs_norm, multi_class='ovr', labels=unique_classes)
            print(f"ROC-AUC (Multi-class): {roc_auc:.4f}")
        else:
            print("ROC-AUC (Multi-class): Cannot compute for single class")
    except Exception as e:
        import traceback
        print("ROC-AUC calculation failed:", e)
        traceback.print_exc()
    
    # Overall accuracy
    accuracy = np.mean(preds == labels)
    print(f"Overall Accuracy: {accuracy:.4f}")

if __name__ == '__main__':
    main()
