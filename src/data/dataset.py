"""
dataset.py
PyTorch Dataset class for loading patches with MONAI transforms
"""

import numpy as np
from pathlib import Path
import torch
from monai.data import Dataset as MonaiDataset
from monai.transforms import (
    Compose,
    ScaleIntensityRanged,
    Lambdad,
    RandFlipd,
    RandRotate90d,
    RandGaussianNoised,
    RandAdjustContrastd,
    ToTensord,
    Lambdad
)


def get_train_transforms(image_min, image_max):
    """
    Transformations for training data
    
    Parameters:
        image_min: Minimum value in images (e.g., 224)
        image_max: Maximum value in images (e.g., 706)
    """
    return Compose([
        # 1. Normalize ONLY the image from [image_min, image_max] to [0, 1]
        ScaleIntensityRanged(
            keys=["image"],
            a_min=image_min,
            a_max=image_max,
            b_min=0.0,
            b_max=1.0,
            clip=True
        ),
        
        # 2. Convert ONLY the mask: 255 -> 1
        Lambdad(
            keys=["mask"],
            func=lambda x: x // 255  # 255 becomes 1, 0 stays 0
        ),
        
        # 3. Flip on Z axis (depth) - applied to BOTH image and mask
        RandFlipd(
            keys=["image", "mask"],
            spatial_axis=0,
            prob=0.5
        ),
        
        # 4. Flip on Y axis (height) - applied to BOTH image and mask
        RandFlipd(
            keys=["image", "mask"],
            spatial_axis=1,
            prob=0.5
        ),
        
        # 5. Flip on X axis (width) - applied to BOTH image and mask
        RandFlipd(
            keys=["image", "mask"],
            spatial_axis=2,
            prob=0.5
        ),
        
        # 6. Random 90-degree rotation - applied to BOTH image and mask
        RandRotate90d(
            keys=["image", "mask"],
            prob=0.5,
            spatial_axes=(0, 1)
        ),
        
        # 7. Add Gaussian noise - applied ONLY to image
        RandGaussianNoised(
            keys=["image"],
            prob=0.3,
            std=0.01
        ),
        
        # 8. Adjust contrast - applied ONLY to image
        RandAdjustContrastd(
            keys=["image"],
            prob=0.3,
            gamma=(0.8, 1.2)
        ),
        
        # 9. Convert both to PyTorch tensors
        ToTensord(keys=["image", "mask"])
    ])


def get_val_transforms(image_min, image_max):
    """
    Transformations for validation/test data (no augmentations)
    """
    return Compose([
        # Normalize ONLY the image
        ScaleIntensityRanged(
            keys=["image"],
            a_min=image_min,
            a_max=image_max,
            b_min=0.0,
            b_max=1.0,
            clip=True
        ),
        
        # Convert ONLY the mask: 255 -> 1
        Lambdad(
            keys=["mask"],
            func=lambda x: x // 255
        ),
        
        # Convert both to PyTorch tensors
        ToTensord(keys=["image", "mask"])
    ])


def create_dataset(data_dir, split_file, transforms, image_min, image_max):
    """
    Create a MONAI Dataset from a split file
    """
    # Read split file (list of patch names)
    with open(split_file, 'r') as f:
        patch_names = [line.strip() for line in f.readlines()]
    
    # Create data list for MONAI
    data_list = []
    image_dir = Path(data_dir) / "images"
    mask_dir = Path(data_dir) / "masks"
    
    for name in patch_names:
        data_list.append({
            "image": str(image_dir / f"{name}.npy"),
            "mask": str(mask_dir / f"{name}.npy")
        })
    
    # Create dataset with transforms
    dataset = MonaiDataset(
        data=data_list,
        transform=transforms(image_min, image_max)
    )
    
    return dataset


def create_datasets(data_dir, splits_dir, batch_size=8, num_workers=4):
    """
    Create train, val, and test datasets with their respective transforms
    """
    from torch.utils.data import DataLoader
    
    splits_dir = Path(splits_dir)
    
    # Auto-detect image range from first image
    image_dir = Path(data_dir) / "images"
    first_image = sorted(image_dir.glob("*.npy"))[0]
    sample_image = np.load(first_image)
    image_min = float(sample_image.min())
    image_max = float(sample_image.max())
    
    print(f"Auto-detected image range: [{image_min:.2f}, {image_max:.2f}]")
    print(f"Formula: (x - {image_min:.2f}) / ({image_max:.2f} - {image_min:.2f})")
    print()
    
    # Create datasets
    print("Creating datasets...")
    
    train_dataset = create_dataset(
        data_dir=data_dir,
        split_file=splits_dir / "train.txt",
        transforms=get_train_transforms,
        image_min=image_min,
        image_max=image_max
    )
    
    val_dataset = create_dataset(
        data_dir=data_dir,
        split_file=splits_dir / "val.txt",
        transforms=get_val_transforms,
        image_min=image_min,
        image_max=image_max
    )
    
    test_dataset = create_dataset(
        data_dir=data_dir,
        split_file=splits_dir / "test.txt",
        transforms=get_val_transforms,
        image_min=image_min,
        image_max=image_max
    )
    
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")
    print(f"Test dataset: {len(test_dataset)} samples")
    print()
    
    # Create DataLoaders
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