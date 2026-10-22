#!/usr/bin/env python
"""
Test script for simple dataset without MONAI
"""

import sys
from pathlib import Path

project_root = Path("/mnt/x/segmentation_cfos")
sys.path.insert(0, str(project_root))

import torch
from src.data.dataset_simple import create_datasets_simple

def main():
    data_dir = project_root / "data" / "raw" / "patches_64" / "useful_patches"
    splits_dir = project_root / "data" / "raw" / "patches_64" / "splits"
    
    train_loader, val_loader, test_loader = create_datasets_simple(
        data_dir=data_dir,
        splits_dir=splits_dir,
        batch_size=4,
        num_workers=0  
    )
    
    print("\n" + "=" * 60)
    print("TESTING ONE BATCH")
    print("=" * 60)
    
    for images, masks in train_loader:
        print(f"Images batch shape: {images.shape}")
        print(f"Images dtype: {images.dtype}")
        print(f"Images min: {images.min():.4f}, max: {images.max():.4f}")
        print(f"Masks batch shape: {masks.shape}")
        print(f"Masks dtype: {masks.dtype}")
        print(f"Masks unique values: {torch.unique(masks)}")
        print(f"Masks min: {masks.min():.4f}, max: {masks.max():.4f}")
        
        if torch.isnan(images).any():
            print("❌ ERROR: NaN values detected in images!")
        elif torch.isinf(images).any():
            print("❌ ERROR: Inf values detected in images!")
        else:
            print("✅ No NaN or Inf values detected")
        
        break
    
    print("\n✅ Dataset test passed!")

if __name__ == "__main__":
    main()