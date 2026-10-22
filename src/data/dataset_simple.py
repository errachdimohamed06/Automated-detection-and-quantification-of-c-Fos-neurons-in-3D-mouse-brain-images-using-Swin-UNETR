"""
dataset_simple.py
PyTorch Dataset without MONAI (compatible with all numpy versions)
"""

import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader


class PatchDataset(Dataset):
    """Simple PyTorch Dataset for loading patches without MONAI"""
    
    def __init__(self, data_dir, split_file, is_train=True):
        """
        Parameters:
            data_dir: Path to directory containing images/ and masks/
            split_file: Path to .txt file with patch names
            is_train: If True, apply augmentations. If False, no augmentations.
        """
        self.data_dir = Path(data_dir)
        self.is_train = is_train
        
        # Read split file
        with open(split_file, 'r') as f:
            self.patch_names = [line.strip() for line in f.readlines() if line.strip()]
    
    def __len__(self):
        return len(self.patch_names)
    
    def _normalize_image(self, image):
        """Min-Max normalization to [0, 1] - per patch"""
        min_val = image.min()
        max_val = image.max()
        
        # Avoid division by zero
        if max_val - min_val == 0:
            return np.zeros_like(image, dtype=np.float32)
        
        return (image.astype(np.float32) - min_val) / (max_val - min_val)
    
    def _convert_mask(self, mask):
        """Convert mask: 255 -> 1, 0 -> 0"""
        return (mask // 255).astype(np.float32)
    
    def _random_flip(self, image, mask, axis):
        """Random flip on a given axis (0=Z, 1=Y, 2=X)"""
        if np.random.random() > 0.5:
            image = np.flip(image, axis=axis).copy()
            mask = np.flip(mask, axis=axis).copy()
        return image, mask
    
    def _random_rotate_90(self, image, mask):
        """Random 90-degree rotation in XY plane"""
        if np.random.random() > 0.5:
            k = np.random.randint(1, 4)
            image = np.rot90(image, k=k, axes=(1, 2)).copy()
            mask = np.rot90(mask, k=k, axes=(1, 2)).copy()
        return image, mask
    
    def _random_gaussian_noise(self, image):
        """Add Gaussian noise to image"""
        if np.random.random() > 0.7:
            noise = np.random.normal(0, 0.01, image.shape).astype(np.float32)
            image = image + noise
            image = np.clip(image, 0, 1)
        return image
    
    def _random_contrast(self, image):
        """Random contrast adjustment using gamma correction"""
        if np.random.random() > 0.7:
            gamma = np.random.uniform(0.8, 1.2)
            # Add a small epsilon to avoid power of zero
            epsilon = 1e-6
            image = np.power(image + epsilon, gamma)
            image = np.clip(image, 0, 1)
        return image
    
    def __getitem__(self, idx):
        # Load image and mask
        name = self.patch_names[idx]
        image = np.load(self.data_dir / "images" / f"{name}.npy")
        mask = np.load(self.data_dir / "masks" / f"{name}.npy")
        
        # Normalize image to [0, 1] (per patch)
        image = self._normalize_image(image)
        
        # Convert mask: 255 -> 1
        mask = self._convert_mask(mask)
        
        # Add channel dimension for PyTorch (1, 64, 64, 64)
        image = image[np.newaxis, ...]
        mask = mask[np.newaxis, ...]
        
        # Apply augmentations only for training
        if self.is_train:
            # Random flips
            image, mask = self._random_flip(image, mask, axis=0)  # Z axis
            image, mask = self._random_flip(image, mask, axis=1)  # Y axis
            image, mask = self._random_flip(image, mask, axis=2)  # X axis
            
            # Random rotation
            image, mask = self._random_rotate_90(image, mask)
            
            # Random noise and contrast (image only)
            image = self._random_gaussian_noise(image)
            image = self._random_contrast(image)
        
        # Convert to torch tensors
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()
        
        return image, mask


def create_datasets_simple(data_dir, splits_dir, batch_size=8, num_workers=0):
    """
    Create train, val, and test datasets without MONAI
    
    Note: num_workers=0 is recommended to avoid multiprocessing issues
    """
    splits_dir = Path(splits_dir)
    
    print("=" * 60)
    print("CREATING DATASETS (Simple version without MONAI)")
    print("=" * 60)
    
    # Create datasets
    train_dataset = PatchDataset(
        data_dir=data_dir,
        split_file=splits_dir / "train.txt",
        is_train=True
    )
    
    val_dataset = PatchDataset(
        data_dir=data_dir,
        split_file=splits_dir / "val.txt",
        is_train=False
    )
    
    test_dataset = PatchDataset(
        data_dir=data_dir,
        split_file=splits_dir / "test.txt",
        is_train=False
    )
    
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")
    print(f"Test dataset: {len(test_dataset)} samples")
    print()
    
    # Create DataLoaders with num_workers=0 to avoid multiprocessing issues
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader