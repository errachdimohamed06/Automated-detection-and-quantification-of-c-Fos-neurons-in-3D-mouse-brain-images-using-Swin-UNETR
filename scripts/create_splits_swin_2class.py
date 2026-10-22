#!/usr/bin/env python
"""
Crée les splits train/val/test pour les patches 2 classes - Swin UNETR
"""

import sys
import random
from pathlib import Path

project_root = Path("/mnt/x/segmentation_cfos")
sys.path.insert(0, str(project_root))


def create_splits_swin(data_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    data_dir = Path(data_dir)
    splits_dir = data_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    mask_dir = data_dir / "masks"
    mask_files = sorted(mask_dir.glob("*.npy"))
    patch_names = [f.stem for f in mask_files]
    total = len(patch_names)

    print("=" * 60)
    print("CRÉATION DES SPLITS (2 CLASSES - SWIN UNETR)")
    print("=" * 60)
    print(f"Total patches : {total}")
    print(f"Train ratio : {train_ratio*100:.0f}%")
    print(f"Val ratio   : {val_ratio*100:.0f}%")
    print(f"Test ratio  : {test_ratio*100:.0f}%")
    print()

    random.seed(seed)
    shuffled = patch_names.copy()
    random.shuffle(shuffled)

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_names = shuffled[:train_end]
    val_names = shuffled[train_end:val_end]
    test_names = shuffled[val_end:]

    print(f"Train : {len(train_names)} patches")
    print(f"Val   : {len(val_names)} patches")
    print(f"Test  : {len(test_names)} patches")
    print()

    with open(splits_dir / "train.txt", 'w') as f:
        f.write('\n'.join(train_names))

    with open(splits_dir / "val.txt", 'w') as f:
        f.write('\n'.join(val_names))

    with open(splits_dir / "test.txt", 'w') as f:
        f.write('\n'.join(test_names))

    print(f"✅ Splits sauvegardés dans : {splits_dir}")
    return train_names, val_names, test_names


if __name__ == "__main__":
    data_dir = Path("/mnt/x/donnees/pachs_64_2classes_slices40")
    create_splits_swin(data_dir)
