import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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

import argparse
import numpy as np
from models.siamese_net import SiameseNetwork
from src.data_loader import get_data_loaders, get_contrastive_data_loader
from utils.losses import NTXentLoss

def train_contrastive(model, data_loader, optimizer, criterion, device, epochs):
    """Train the model with self-supervised contrastive learning using distinct augmented views"""
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        for batch in data_loader:
            view1, view2, _ = batch
            if view1.shape[0] == 0:
                continue
            view1, view2 = view1.to(device), view2.to(device)

            proj1, proj2 = model(view1, view2)

            if torch.isnan(proj1).any() or torch.isnan(proj2).any():
                print(f"Warning: NaN in projections at epoch {epoch+1}, skipping batch")
                optimizer.zero_grad()
                continue

            loss = criterion(proj1, proj2)

            if torch.isnan(loss) or loss.item() > 1e6:
                print(f"Warning: NaN/Inf loss detected at epoch {epoch+1}, skipping batch")
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        avg_loss = epoch_loss / max(batch_count, 1)
        print(f"Contrastive Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}", flush=True)

def train_classification(model, train_loader, val_loader, optimizer, criterion, device, epochs, save_path):
    """Train the model for classification using class weighting and validation tracking"""
    best_val_loss = float('inf')
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        batch_count = 0

        for batch in train_loader:
            images, labels = batch
            if images.shape[0] == 0:
                continue
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                optimizer.zero_grad()
                continue

            loss = criterion(outputs, labels)

            if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 1e6:
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            batch_count += 1

        train_acc = 100. * train_correct / max(train_total, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_batch_count = 0

        with torch.no_grad():
            for batch in val_loader:
                images, labels = batch
                if images.shape[0] == 0:
                    continue
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    continue
                
                loss = criterion(outputs, labels)
                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                val_batch_count += 1

        avg_val_loss = val_loss / max(val_batch_count, 1)
        val_acc = 100. * val_correct / max(val_total, 1)

        print(f"Classification Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/max(batch_count, 1):.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%", flush=True)

        # Save best model based on validation loss / accuracy
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  [SAVED] Best model checkpoint saved to {save_path} (val acc: {val_acc:.2f}%)", flush=True)

def main():
    parser = argparse.ArgumentParser(description='Train Siamese Network for Fatty Liver Classification')
    parser.add_argument('--data_dir', type=str, default='data', help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=30, help='Number of classification epochs')
    parser.add_argument('--contrastive_epochs', type=int, default=10, help='Number of contrastive pre-training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--save_path', type=str, default='best_model.pth', help='Path to save best model')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data with leakage-resistant patient split
    try:
        train_loader, val_loader, test_loader = get_data_loaders(args.data_dir, args.batch_size)
        contrastive_loader = get_contrastive_data_loader(args.data_dir, args.batch_size)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    # Calculate class weights for imbalanced loss
    class_counts = torch.zeros(5)
    for _, labels in train_loader:
        for lbl in labels:
            class_counts[lbl] += 1
    total_samples = class_counts.sum()
    class_weights = total_samples / (5.0 * (class_counts + 1e-5))
    class_weights = class_weights.to(device)

    # Create model
    model = SiameseNetwork().to(device)

    # Contrastive pre-training
    if args.contrastive_epochs > 0:
        print("Starting contrastive pre-training...")
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        criterion = NTXentLoss(temperature=0.5)
        train_contrastive(model, contrastive_loader, optimizer, criterion, device, args.contrastive_epochs)

    # Classification training
    print("Starting classification training...")
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    train_classification(model, train_loader, val_loader, optimizer, criterion, device, args.epochs, args.save_path)

    # Final evaluation on held-out test dataset
    print("\n🔍 Evaluating best model on held-out test dataset...")
    if Path(args.save_path).exists():
        model.load_state_dict(torch.load(args.save_path, map_location=device))
    model.eval()
    test_correct = 0
    test_total = 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    test_acc = 100. * test_correct / max(test_total, 1)
    print(f"✅ Final Test Accuracy: {test_acc:.2f}% ({test_correct}/{test_total} test scans)")

    class_names = ['Normal', 'Grade-I', 'Grade-II', 'Grade-III', 'CLD']
    all_preds, all_targets = np.array(all_preds), np.array(all_targets)
    for c_idx, c_name in enumerate(class_names):
        mask = (all_targets == c_idx)
        if mask.sum() > 0:
            c_acc = (all_preds[mask] == all_targets[mask]).mean() * 100
            print(f"   {c_name:10} | Test Accuracy: {c_acc:6.2f}% | Samples: {mask.sum():2}")

    print("\nTraining completed successfully!")

if __name__ == '__main__':
    main()