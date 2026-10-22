#!/usr/bin/env python
"""
Script to create train/val/test splits for the useful patches
Simple random split without grouping by original image
"""

import sys
from pathlib import Path
import random
import shutil

project_root = Path("/mnt/x/segmentation_cfos")
sys.path.insert(0, str(project_root))

def create_splits():
    # Paths
    useful_image_dir = project_root / "data" / "raw" / "patches_64" / "useful_patches" / "images"
    useful_mask_dir = project_root / "data" / "raw" / "patches_64" / "useful_patches" / "masks"
    
    splits_dir = project_root / "data" / "raw" / "patches_64" / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    image_files = sorted(useful_image_dir.glob("*.npy"))
    patch_names = [f.stem for f in image_files]  
    
    total = len(patch_names)
    print("=" * 60)
    print("CREATING TRAIN/VAL/TEST SPLITS")
    print("=" * 60)
    print(f"Total useful patches: {total}")
    print(f"Splits directory: {splits_dir}")
    print()
    
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    
    random.seed(42) 
    shuffled = patch_names.copy()
    random.shuffle(shuffled)
    
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_names = shuffled[:train_end]
    val_names = shuffled[train_end:val_end]
    test_names = shuffled[val_end:]
    
    print(f"Train patches: {len(train_names)} ({len(train_names)/total*100:.1f}%)")
    print(f"Val patches: {len(val_names)} ({len(val_names)/total*100:.1f}%)")
    print(f"Test patches: {len(test_names)} ({len(test_names)/total*100:.1f}%)")
    print()
    
    train_path = splits_dir / "train.txt"
    val_path = splits_dir / "val.txt"
    test_path = splits_dir / "test.txt"
    
    with open(train_path, 'w') as f:
        f.write('\n'.join(train_names))
    
    with open(val_path, 'w') as f:
        f.write('\n'.join(val_names))
    
    with open(test_path, 'w') as f:
        f.write('\n'.join(test_names))
    
    print("Files saved:")
    print(f"  {train_path}")
    print(f"  {val_path}")
    print(f"  {test_path}")
    print()
    
    print("First 10 train patches:")
    for name in train_names[:10]:
        print(f"  {name}")
    
    print("\nFirst 10 val patches:")
    for name in val_names[:10]:
        print(f"  {name}")
    
    print("\nFirst 10 test patches:")
    for name in test_names[:10]:
        print(f"  {name}")
    
    print("\n" + "=" * 60)
    print("SPLITS CREATED SUCCESSFULLY!")
    print("=" * 60)
    
    return train_names, val_names, test_names

if __name__ == "__main__":
    create_splits()