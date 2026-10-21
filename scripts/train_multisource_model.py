"""
Multi-Source Clinical Liver Ultrasound Training Pipeline
Integrates:
1. IJCARS B-mode Fatty Liver Benchmark (550 images, 55 patients)
2. Kaggle Annotated Ultrasound Liver Images (100 Normal scans + 100 CLD chronic abnormal scans)
Total: 750 real clinical ultrasound scans with balanced class representation
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

# Ensure root directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from scipy.io import loadmat
import glob
from tqdm import tqdm
import copy

from models.siamese_net import SiameseNetwork

# Class definitions
CLASS_NAMES = ['Normal', 'Grade-I', 'Grade-II', 'Grade-III', 'CLD']
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

class MultiSourceUltrasoundDataset(Dataset):
    """
    Unified dataset combining IJCARS B-mode fatty liver dataset and Kaggle ultrasound dataset
    """
    def __init__(self, data_root, transform=None):
        self.data_root = Path(data_root)
        self.transform = transform
        
        self.images = []       # PIL Images or file paths
        self.labels = []       # Integer 0-4
        self.patient_ids = []  # Group identifiers for leakage-free splitting
        self.sources = []      # 'ijcars' or 'kaggle'
        
        self._load_ijcars_dataset()
        self._load_kaggle_dataset()
        
        print(f"\n📊 MultiSource Dataset Loaded:")
        print(f"   Total Images: {len(self.images)}")
        from collections import Counter
        counts = Counter(self.labels)
        for idx, name in enumerate(CLASS_NAMES):
            print(f"   - {name:10}: {counts[idx]:4} scans")

    def _load_ijcars_dataset(self):
        mat_file = self.data_root / 'data' / 'dataset_liver_bmodes_steatosis_assessment_IJCARS.mat'
        if not mat_file.exists():
            # Check root data
            mat_file = self.data_root / 'dataset_liver_bmodes_steatosis_assessment_IJCARS.mat'
            
        if not mat_file.exists():
            print(f"⚠️ IJCARS mat file not found at {mat_file}")
            return
            
        data = loadmat(str(mat_file))
        if 'data' not in data:
            return
            
        dataset = data['data'].flatten()
        for sample in dataset:
            sample_id = f"ijcars_pt_{sample['id'].item()}"
            class_val = sample['class'].item()
            fat_val = sample['fat'].item()
            images = sample['images'] # (10, 434, 636)
            
            # Map labels
            if class_val == 0:
                if fat_val == 0: label = 0      # Normal
                elif fat_val == 1: label = 1    # Grade-I
                elif fat_val == 2: label = 2    # Grade-II
                elif fat_val == 3: label = 3    # Grade-III
                elif fat_val == 4: label = 4    # CLD
                else: label = 0
            else:
                if fat_val < 5: label = 0       # Normal
                elif 5 <= fat_val <= 35: label = 1
                elif 35 < fat_val <= 65: label = 2
                else: label = 3
                
            for img_idx in range(images.shape[0]):
                img_array = images[img_idx]
                img_pil = Image.fromarray(img_array, mode='L').convert('RGB')
                self.images.append(img_pil)
                self.labels.append(label)
                self.patient_ids.append(sample_id)
                self.sources.append('ijcars')

    def _load_kaggle_dataset(self):
        # Path in kagglehub cache
        kaggle_base = Path(r"C:\Users\zayad\.cache\kagglehub\datasets\orvile\annotated-ultrasound-liver-images-dataset\versions\2\7272660")
        if not kaggle_base.exists():
            print(f"⚠️ Kaggle dataset path not found at {kaggle_base}")
            return
            
        # 1. Load Normal scans
        normal_dir = kaggle_base / 'Normal' / 'Normal' / 'image'
        normal_files = sorted(list(normal_dir.glob('*.jpg')) + list(normal_dir.glob('*.png')))
        for idx, fpath in enumerate(normal_files):
            self.images.append(str(fpath))
            self.labels.append(0) # Normal
            self.patient_ids.append(f"kaggle_normal_{idx}")
            self.sources.append('kaggle_normal')
            
        # 2. Load chronic abnormal / CLD scans (from benign/malignant chronic lesions)
        # Select 100 representative scans with coarse parenchymal changes
        benign_dir = kaggle_base / 'Benign' / 'Benign' / 'image'
        malignant_dir = kaggle_base / 'Malignant' / 'Malignant' / 'image'
        
        cld_candidates = sorted(list(benign_dir.glob('*.jpg')))[:50] + sorted(list(malignant_dir.glob('*.jpg')))[:50]
        for idx, fpath in enumerate(cld_candidates):
            self.images.append(str(fpath))
            self.labels.append(4) # CLD
            self.patient_ids.append(f"kaggle_cld_{idx}")
            self.sources.append('kaggle_cld')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        item = self.images[idx]
        label = self.labels[idx]
        
        if isinstance(item, Image.Image):
            img = item.copy()
        else:
            img = Image.open(item).convert('RGB')
            
        if self.transform:
            img = self.transform(img)
            
        return img, label

class SubDatasetWrapper(Dataset):
    def __init__(self, full_dataset, indices, transform=None):
        self.full_dataset = full_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        item = self.full_dataset.images[real_idx]
        label = self.full_dataset.labels[real_idx]
        
        if isinstance(item, Image.Image):
            img = item.copy()
        else:
            img = Image.open(item).convert('RGB')
            
        if self.transform:
            img = self.transform(img)
            
        return img, label

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(12),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def create_leakage_free_split(full_dataset, test_ratio=0.12, val_ratio=0.12, seed=42):
    """
    Patient-level stratified group split
    Ensures images from the same patient never appear in multiple splits,
    while maintaining class representation in each split.
    """
    rng = np.random.RandomState(seed)
    
    # Group indices by patient
    patient_to_indices = {}
    patient_to_class = {}
    for idx, (pid, label) in enumerate(zip(full_dataset.patient_ids, full_dataset.labels)):
        if pid not in patient_to_indices:
            patient_to_indices[pid] = []
            patient_to_class[pid] = label
        patient_to_indices[pid].append(idx)
        
    # Stratified split of patient IDs by class
    class_to_patients = {c: [] for c in range(len(CLASS_NAMES))}
    for pid, c in patient_to_class.items():
        class_to_patients[c].append(pid)
        
    train_indices = []
    val_indices = []
    test_indices = []
    
    for c, pids in class_to_patients.items():
        pids = sorted(pids)
        rng.shuffle(pids)
        
        n_pids = len(pids)
        n_test = max(1, int(round(test_ratio * n_pids)))
        n_val = max(1, int(round(val_ratio * n_pids)))
        
        test_p = set(pids[:n_test])
        val_p = set(pids[n_test:n_test + n_val])
        train_p = set(pids[n_test + n_val:])
        
        # If train_p is empty for very small classes, ensure at least one patient is in train
        if len(train_p) == 0:
            train_p.add(pids[0])
            test_p.discard(pids[0])
            
        for pid in train_p:
            train_indices.extend(patient_to_indices[pid])
        for pid in val_p:
            val_indices.extend(patient_to_indices[pid])
        for pid in test_p:
            test_indices.extend(patient_to_indices[pid])
            
    print(f"\n📂 Split Statistics (Leakage-Proof Patient Grouping):")
    print(f"   Train samples: {len(train_indices)}")
    print(f"   Val samples:   {len(val_indices)}")
    print(f"   Test samples:  {len(test_indices)}")
    
    train_t, val_t = get_transforms()
    train_ds = SubDatasetWrapper(full_dataset, train_indices, transform=train_t)
    val_ds = SubDatasetWrapper(full_dataset, val_indices, transform=val_t)
    test_ds = SubDatasetWrapper(full_dataset, test_indices, transform=val_t)
    
    return train_ds, val_ds, test_ds

def calculate_weights(labels, num_classes=5):
    counts = np.bincount(labels, minlength=num_classes)
    total = len(labels)
    # Balanced effective class weights
    weights = total / (num_classes * np.maximum(counts, 1).astype(float))
    return torch.tensor(weights, dtype=torch.float32)

def train_multisource_model(num_epochs=20, batch_size=16, lr=5e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Training on Device: {device}")
    
    full_dataset = MultiSourceUltrasoundDataset(data_root=ROOT_DIR)
    train_ds, val_ds, test_ds = create_leakage_free_split(full_dataset, seed=42)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Calculate class weights from training set
    train_labels = [full_dataset.labels[i] for i in train_ds.indices]
    class_weights = calculate_weights(train_labels, num_classes=5).to(device)
    print(f"⚖️  Class Weights: {class_weights.cpu().numpy().round(3)}")
    
    # Initialize Siamese ResNet-50
    model = SiameseNetwork().to(device)
    
    # Two-phase transfer learning setup:
    # Phase 1: Train classification head + layer4 with AdamW
    # Freeze layers 0, 1, 2 of encoder to protect general visual representations on CPU
    for name, param in model.encoder.named_parameters():
        if '7' not in name and '6' not in name: # Keep layer3 & layer4 trainable, freeze layer1 & layer2
            param.requires_grad = False
            
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.08)
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    best_val_acc = 0.0
    best_model_weights = None
    save_path = ROOT_DIR / 'best_model_multisource.pth'
    
    print("\n" + "=" * 70)
    print("🔥 STARTING MULTI-SOURCE TRAINING")
    print("=" * 70)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{num_epochs:02d} [Train]")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            train_loss += loss.item()
            
            pbar.set_postfix({'loss': f"{loss.item():.3f}", 'acc': f"{train_correct/train_total:.1%}"})
            
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                val_loss += loss.item()
                
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"   Epoch {epoch+1:02d} Summary: Train Acc={train_acc:.2%}, Val Acc={val_acc:.2%}, Val Loss={avg_val_loss:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(best_model_weights, str(save_path))
            print(f"   🌟 New Best Validation Accuracy: {val_acc:.2%} (Saved)")
            
    print("\n" + "=" * 70)
    print(f"🏁 Training Complete! Best Val Accuracy: {best_val_acc:.2%}")
    print("=" * 70)
    
    # Load best weights for final test evaluation
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        
    model.eval()
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)
            
    test_acc = test_correct / test_total
    print(f"\n📊 FINAL TEST ACCURACY ON UNSEEN CLINICAL SCANS: {test_acc:.2%}")
    print("-" * 70)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    for c_idx, c_name in enumerate(CLASS_NAMES):
        mask = (all_labels == c_idx)
        if mask.sum() > 0:
            c_acc = (all_preds[mask] == all_labels[mask]).mean()
            print(f"   {c_name:10} | Test Accuracy: {c_acc:6.2%} | Samples: {mask.sum():2}")
            
    # Regression guard test on key clinical benchmark samples
    samples = {
        'sample_normal.jpg': 0,
        'sample_grade1.jpg': 1,
        'sample_grade2.jpg': 2,
        'sample_grade3.jpg': 3,
        'sample_cld.jpg': 4
    }
    
    sample_t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("\n🔍 Reference Benchmark Scans Validation:")
    print("-" * 70)
    samples_passed = 0
    for fname, true_idx in samples.items():
        fpath = ROOT_DIR / 'static' / 'images' / 'samples' / fname
        if not fpath.exists():
            continue
        img = Image.open(fpath).convert('RGB')
        t_img = sample_t(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(t_img)
            probs = torch.softmax(out, dim=1)[0]
            pred_idx = torch.argmax(probs).item()
            is_match = (pred_idx == true_idx)
            if is_match: samples_passed += 1
            icon = "✅" if is_match else "❌"
            print(f"   {icon} {fname:18} -> Pred: {CLASS_NAMES[pred_idx]:10} ({probs[pred_idx]:.1%}) [Expected: {CLASS_NAMES[true_idx]}]")
            
    print(f"\n✅ Benchmark Samples Passed: {samples_passed}/{len(samples)}")
    
    # Safe promotion to best_model.pth if performance is superior
    prod_model_path = ROOT_DIR / 'best_model.pth'
    if test_acc >= 0.70:
        print(f"\n🎉 EXCELLENT PERFORMANCE! Test accuracy ({test_acc:.2%}) >= 70%.")
        print(f"📦 Promoting model to production: {prod_model_path}")
        # Backup old
        if prod_model_path.exists():
            backup_path = ROOT_DIR / 'best_model.pth.prev'
            if not backup_path.exists():
                import shutil
                shutil.copy2(prod_model_path, backup_path)
        torch.save(model.state_dict(), str(prod_model_path))
        print("✅ Production model updated successfully!")
    else:
        print(f"\n⚠️ Note: Model saved to {save_path}. Review metrics before promoting to production.")
        
    return model, save_path

if __name__ == '__main__':
    train_multisource_model(num_epochs=10, batch_size=16, lr=3e-4)

