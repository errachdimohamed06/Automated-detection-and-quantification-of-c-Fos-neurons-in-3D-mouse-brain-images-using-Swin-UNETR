#!/usr/bin/env python
"""
Crée les splits train/val/test pour les patches 3 classes
80% train, 10% val, 10% test
"""

import sys
import random
from pathlib import Path

project_root = Path("/mnt/x/segmentation_cfos")
sys.path.insert(0, str(project_root))


def create_splits(data_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Crée les fichiers split.txt pour train/val/test

    data_dir : dossier contenant les dossiers images/ et masks/
    train_ratio : proportion pour l'entraînement (0.8)
    val_ratio : proportion pour la validation (0.1)
    test_ratio : proportion pour le test (0.1)
    seed : graine aléatoire pour reproductibilité
    """
    data_dir = Path(data_dir)
    splits_dir = data_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    image_dir = data_dir / "images"
    image_files = sorted(image_dir.glob("*.npy"))

    if len(image_files) == 0:
        image_files = sorted(image_dir.glob("*.tif"))

    patch_names = [f.stem for f in image_files]
    total = len(patch_names)

    print("=" * 60)
    print("CRÉATION DES SPLITS (3 CLASSES)")
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

    train_path = splits_dir / "train.txt"
    val_path = splits_dir / "val.txt"
    test_path = splits_dir / "test.txt"

    with open(train_path, "w") as f:
        f.write("\n".join(train_names))

    with open(val_path, "w") as f:
        f.write("\n".join(val_names))

    with open(test_path, "w") as f:
        f.write("\n".join(test_names))

    print(f"✅ Splits sauvegardés dans : {splits_dir}")
    print(f"   - train.txt : {len(train_names)} patches")
    print(f"   - val.txt   : {len(val_names)} patches")
    print(f"   - test.txt  : {len(test_names)} patches")
    print("=" * 60)

    return train_names, val_names, test_names


if __name__ == "__main__":
    data_dir = Path("/mnt/x/donnees/pachs_64_3classes_filtered")
    create_splits(data_dir)