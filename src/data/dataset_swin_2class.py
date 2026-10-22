"""
dataset_swin_2class.py
Dataset pour Swin UNETR - 2 classes (fond, neurones+vaisseaux)
Fichier dédié pour éviter les conflits avec U-Net
"""

import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader


class PatchDatasetSwin2Class(Dataset):
    """
    Dataset pour patches 2 classes (Swin UNETR)
    """

    def __init__(self, data_dir, split_file, is_train=True):
        self.data_dir = Path(data_dir)
        self.is_train = is_train

        with open(split_file, 'r') as f:
            self.patch_names = [line.strip() for line in f.readlines() if line.strip()]

    def __len__(self):
        return len(self.patch_names)

    def _normalize_image(self, image):
        min_val = image.min()
        max_val = image.max()
        if max_val - min_val == 0:
            return np.zeros_like(image, dtype=np.float32)
        return (image.astype(np.float32) - min_val) / (max_val - min_val)

    def _random_flip(self, image, mask, axis):
        if np.random.random() > 0.5:
            image = np.flip(image, axis=axis).copy()
            mask = np.flip(mask, axis=axis).copy()
        return image, mask

    def _random_rotate_90(self, image, mask):
        if np.random.random() > 0.5:
            k = np.random.randint(1, 4)
            image = np.rot90(image, k=k, axes=(1, 2)).copy()
            mask = np.rot90(mask, k=k, axes=(1, 2)).copy()
        return image, mask

    def _random_gaussian_noise(self, image):
        if np.random.random() > 0.7:
            noise = np.random.normal(0, 0.01, image.shape).astype(np.float32)
            image = image + noise
            image = np.clip(image, 0, 1)
        return image

    def _random_contrast(self, image):
        if np.random.random() > 0.7:
            gamma = np.random.uniform(0.8, 1.2)
            epsilon = 1e-6
            image = np.power(image + epsilon, gamma)
            image = np.clip(image, 0, 1)
        return image

    def __getitem__(self, idx):
        name = self.patch_names[idx]

        image_path = self.data_dir / "images" / f"{name}.tif"
        mask_path = self.data_dir / "masks" / f"{name}.npy"

        from src.data.io_utils import read_tif_image, convert_image_to_array

        image_sitk = read_tif_image(str(image_path))
        image = convert_image_to_array(image_sitk)
        mask = np.load(mask_path)

        image = self._normalize_image(image)
        mask = mask.astype(np.float32)

        image = image[np.newaxis, ...]
        mask = mask[np.newaxis, ...]

        if self.is_train:
            image, mask = self._random_flip(image, mask, axis=0)
            image, mask = self._random_flip(image, mask, axis=1)
            image, mask = self._random_flip(image, mask, axis=2)
            image, mask = self._random_rotate_90(image, mask)
            image = self._random_gaussian_noise(image)
            image = self._random_contrast(image)

        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()

        return image, mask


def create_datasets_swin_2class(data_dir, splits_dir, batch_size=8, num_workers=2):
    """
    Crée les 3 datasets (train, val, test) pour Swin UNETR
    """

    splits_dir = Path(splits_dir)
    data_dir = Path(data_dir)

    if not (splits_dir / "train.txt").exists():
        import sys
        project_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(project_root))
        from scripts.create_splits_swin_2class import create_splits_swin
        create_splits_swin(data_dir)

    print("=" * 60)
    print("CREATING DATASETS (2 CLASSES - SWIN UNETR)")
    print("=" * 60)

    train_dataset = PatchDatasetSwin2Class(
        data_dir=data_dir,
        split_file=splits_dir / "train.txt",
        is_train=True
    )

    val_dataset = PatchDatasetSwin2Class(
        data_dir=data_dir,
        split_file=splits_dir / "val.txt",
        is_train=False
    )

    test_dataset = PatchDatasetSwin2Class(
        data_dir=data_dir,
        split_file=splits_dir / "test.txt",
        is_train=False
    )

    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")
    print(f"Test dataset: {len(test_dataset)} samples")
    print()

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
