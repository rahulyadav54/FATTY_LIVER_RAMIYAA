"""
Script to evaluate model confidence calibration, accuracy, ECE (Expected Calibration Error),
and determine validation-derived low-confidence threshold.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F
import numpy as np
from src.data_loader import get_data_loaders
from models.siamese_net import SiameseNetwork

def compute_ece(probs, labels, n_bins=10):
    """
    Compute Expected Calibration Error (ECE)
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    bin_details = []
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
            bin_details.append({
                'bin': f"({bin_lower:.1f}, {bin_upper:.1f}]",
                'count': np.sum(in_bin),
                'accuracy': accuracy_in_bin,
                'confidence': avg_confidence_in_bin
            })
            
    return ece, bin_details

def analyze_dataset_split(model, loader, device, name="Validation"):
    model.eval()
    all_probs = []
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
            
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    confidences = np.max(all_probs, axis=1)
    
    acc = np.mean(all_preds == all_labels)
    ece, bin_details = compute_ece(all_probs, all_labels)
    
    print(f"\n==========================================")
    print(f"📊 {name.upper()} SET ANALYSIS")
    print(f"==========================================")
    print(f"Sample Count:        {len(all_labels)}")
    print(f"Overall Accuracy:    {acc * 100:.2f}%")
    print(f"Expected Calib Error:{ece * 100:.2f}%")
    print(f"Mean Confidence:     {np.mean(confidences) * 100:.2f}%")
    print(f"Min Confidence:      {np.min(confidences) * 100:.2f}%")
    print(f"Max Confidence:      {np.max(confidences) * 100:.2f}%")
    print(f"Median Confidence:   {np.median(confidences) * 100:.2f}%")
    
    correct_mask = (all_preds == all_labels)
    incorrect_mask = ~correct_mask
    
    if np.sum(correct_mask) > 0:
        print(f"Mean Conf (Correct):  {np.mean(confidences[correct_mask]) * 100:.2f}%")
    if np.sum(incorrect_mask) > 0:
        print(f"Mean Conf (Incorrect):{np.mean(confidences[incorrect_mask]) * 100:.2f}%")
        
    return all_probs, all_labels, all_preds, confidences

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating model confidence calibration on device: {device}")
    
    train_loader, val_loader, test_loader = get_data_loaders(data_dir='data', batch_size=16)
    
    model = SiameseNetwork().to(device)
    model_path = 'best_model.pth'
    if Path(model_path).exists():
        state = torch.load(model_path, map_location=device)
        if isinstance(state, dict) and 'model' in state:
            model.load_state_dict(state['model'])
        else:
            model.load_state_dict(state)
        print(f"Loaded model weights from {model_path}")
    else:
        print(f"WARNING: {model_path} not found!")
        
    # Analyze Validation Set
    val_probs, val_labels, val_preds, val_confs = analyze_dataset_split(model, val_loader, device, name="Validation")
    
    # Analyze Test Set
    test_probs, test_labels, test_preds, test_confs = analyze_dataset_split(model, test_loader, device, name="Test")
    
    # Determine confidence threshold on validation set
    # Rationale: For a 5-class problem, uniform random probability is 1/5 = 0.20 (20%).
    # We analyze accuracy as a function of confidence threshold on validation set.
    print("\n------------------------------------------")
    print("🎯 VALIDATION THRESHOLD DETERMINATION")
    print("------------------------------------------")
    percentiles = [10, 20, 25, 30, 40, 50, 60, 70, 80, 90]
    for p in percentiles:
        thresh = np.percentile(val_confs, p)
        high_conf_mask = val_confs >= thresh
        if np.sum(high_conf_mask) > 0:
            acc_at_thresh = np.mean(val_preds[high_conf_mask] == val_labels[high_conf_mask])
            coverage = np.mean(high_conf_mask)
            print(f"Threshold (P{p:02d}): {thresh*100:5.2f}% | Coverage: {coverage*100:5.1f}% | Acc above threshold: {acc_at_thresh*100:5.2f}%")

if __name__ == '__main__':
    main()

