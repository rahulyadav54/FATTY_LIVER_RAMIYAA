import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path
import numpy as np
from scipy.io import loadmat

class FattyLiverDataset(Dataset):
    def __init__(self, data_dir, transform=None, is_train=True):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.is_train = is_train

        # Class mapping
        self.class_names = ['Normal', 'Grade-I', 'Grade-II', 'Grade-III', 'CLD']
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        # Load all image paths, labels, and patient identifiers
        self.image_paths = []
        self.labels = []
        self.patient_ids = []

        # Check if .mat file exists
        mat_file = self.data_dir / 'dataset_liver_bmodes_steatosis_assessment_IJCARS.mat'
        if mat_file.exists():
            # Load from .mat file
            self.load_from_mat(mat_file)
        else:
            # Load from image folders
            for class_name in self.class_names:
                class_dir = self.data_dir / class_name
                if class_dir.exists():
                    for idx, img_path in enumerate(sorted(class_dir.glob('*.jpg'))):
                        self.image_paths.append(img_path)
                        self.labels.append(self.class_to_idx[class_name])
                        # Synthetic/folder patient ID based on index
                        self.patient_ids.append(f"{class_name}_{idx // 10}")

    def load_from_mat(self, mat_file):
        """Load dataset from MATLAB .mat file with verified label mapping"""
        data = loadmat(str(mat_file))

        if 'data' in data:
            dataset = data['data'].flatten()  # Shape (55,)

            for sample in dataset:
                sample_id = sample['id'].item()
                class_val = sample['class'].item()
                fat_val = sample['fat'].item()
                images = sample['images']  # Shape (10, 434, 636)

                # Correct IJCARS label mapping:
                # class_val == 0: Steatosis cohort with integer fat grades (0=Normal, 1=Grade-I, 2=Grade-II, 3=Grade-III, 4=CLD)
                # class_val == 1: Fibrosis cohort with fat percentage values
                if class_val == 0:
                    if fat_val == 0:
                        label = 0  # Normal
                    elif fat_val == 1:
                        label = 1  # Grade-I
                    elif fat_val == 2:
                        label = 2  # Grade-II
                    elif fat_val == 3:
                        label = 3  # Grade-III
                    elif fat_val == 4:
                        label = 4  # CLD
                    else:
                        label = 0
                else:
                    if fat_val < 5:
                        label = 0  # Normal
                    elif 5 <= fat_val <= 35:
                        label = 1  # Grade-I
                    elif 35 < fat_val <= 65:
                        label = 2  # Grade-II
                    else:
                        label = 3  # Grade-III

                # For each of the 10 images in this sample
                for img_idx in range(images.shape[0]):
                    img = images[img_idx]  # Shape (434, 636)
                    img_pil = Image.fromarray(img, mode='L').convert('RGB')
                    self.image_paths.append(img_pil)
                    self.labels.append(label)
                    self.patient_ids.append(sample_id)
        else:
            raise ValueError("MAT file must contain 'data' field with structured array")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = self.image_paths[idx]
        label = self.labels[idx]

        if isinstance(img, Image.Image):
            image = img.copy()
        else:
            image = Image.open(img).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


class FattyLiverSubsetDataset(Dataset):
    """Subset wrapper that applies dedicated transform to selected indices"""
    def __init__(self, full_dataset, indices, transform=None):
        self.full_dataset = full_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img = self.full_dataset.image_paths[real_idx]
        label = self.full_dataset.labels[real_idx]

        if isinstance(img, Image.Image):
            image = img.copy()
        else:
            image = Image.open(img).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


def get_data_transforms(is_train=True):
    """
    Enhanced transforms optimized for ultrasound medical imaging
    """
    if is_train:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(15),
            transforms.RandomAffine(
                degrees=0, 
                shear=5, 
                scale=(0.9, 1.1), 
                translate=(0.05, 0.05)
            ),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.05,
                hue=0.02
            ),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.3)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    return transform


def get_data_loaders(data_dir, batch_size=32, num_workers=0, seed=42):
    """
    Returns train, val, test data loaders using a leakage-resistant Patient-Level Group Split
    """
    train_transform = get_data_transforms(is_train=True)
    test_transform = get_data_transforms(is_train=False)
    
    full_dataset = FattyLiverDataset(data_dir, transform=None)
    
    # Patient-level group split (no patient frame leakage across splits)
    patient_ids = np.array(full_dataset.patient_ids)
    unique_pids = np.array(sorted(list(set(patient_ids))))
    
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_pids)
    
    n_test = max(1, int(0.1 * len(unique_pids)))
    n_val = max(1, int(0.1 * len(unique_pids)))
    
    test_pids = set(unique_pids[:n_test])
    val_pids = set(unique_pids[n_test:n_test + n_val])
    train_pids = set(unique_pids[n_test + n_val:])
    
    train_indices = [i for i, pid in enumerate(patient_ids) if pid in train_pids]
    val_indices = [i for i, pid in enumerate(patient_ids) if pid in val_pids]
    test_indices = [i for i, pid in enumerate(patient_ids) if pid in test_pids]
    
    train_dataset = FattyLiverSubsetDataset(full_dataset, train_indices, transform=train_transform)
    val_dataset = FattyLiverSubsetDataset(full_dataset, val_indices, transform=test_transform)
    test_dataset = FattyLiverSubsetDataset(full_dataset, test_indices, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader


# For contrastive learning (self-supervised)
class ContrastiveDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform or get_data_transforms(is_train=True)
        self.full_dataset = FattyLiverDataset(data_dir, transform=None)

    def __len__(self):
        return len(self.full_dataset)
    
    def __getitem__(self, idx):
        img_raw = self.full_dataset.image_paths[idx]
        label = self.full_dataset.labels[idx]

        if isinstance(img_raw, Image.Image):
            img1 = img_raw.copy()
            img2 = img_raw.copy()
        else:
            img1 = Image.open(img_raw).convert('RGB')
            img2 = Image.open(img_raw).convert('RGB')

        # Generate two distinctly augmented views of the same image
        view1 = self.transform(img1)
        view2 = self.transform(img2)

        return view1, view2, label


def get_contrastive_data_loader(data_dir, batch_size=32, num_workers=0):
    transform = get_data_transforms(is_train=True)
    dataset = ContrastiveDataset(data_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return data_loader
