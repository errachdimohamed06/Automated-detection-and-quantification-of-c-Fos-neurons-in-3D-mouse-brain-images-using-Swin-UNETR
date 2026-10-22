"""
dataset_2class.py
Dataset pour la segmentation binaire (2 classes)
Images: .tif, Masks: .npy
"""

import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from src.data.io_utils import read_tif_image, convert_image_to_array


class PatchDataset2Class(Dataset):
    """
    Dataset pour patches 2 classes
    Images en .tif, Masks en .npy
    """
    
    def __init__(self, data_dir, split_file, is_train=True):
        self.data_dir = Path(data_dir)
        self.is_train = is_train
        
        with open(split_file, 'r') as f:
            self.patch_names = [line.strip() for line in f.readlines() if line.strip()]
        
        print(f"  Dataset chargé: {len(self.patch_names)} patches")
    
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
        
        # Charger l'image (.tif)
        image_path = self.data_dir / "images" / f"{name}.tif"
        image_sitk = read_tif_image(str(image_path))
        image = convert_image_to_array(image_sitk)
        
        # Charger le mask (.npy)
        mask_path = self.data_dir / "masks" / f"{name}.npy"
        mask = np.load(mask_path)
        
        # Normaliser l'image
        image = self._normalize_image(image)
        mask = mask.astype(np.float32)
        
        # Ajouter dimension canal
        image = image[np.newaxis, ...]
        mask = mask[np.newaxis, ...]
        
        # Augmentations (train uniquement)
        if self.is_train:
            image, mask = self._random_flip(image, mask, axis=0)
            image, mask = self._random_flip(image, mask, axis=1)
            image, mask = self._random_flip(image, mask, axis=2)
            image, mask = self._random_rotate_90(image, mask)
            image = self._random_gaussian_noise(image)
            image = self._random_contrast(image)
        
        # Convertir en tenseurs
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).long()
        
        return image, mask


def create_datasets_2class(data_dir, splits_dir, batch_size=8, num_workers=2):
    """
    Crée les 3 datasets avec leurs DataLoaders
    """
    splits_dir = Path(splits_dir)
    data_dir = Path(data_dir)
    
    # Créer les splits si inexistants
    if not (splits_dir / "train.txt").exists():
        import sys
        project_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(project_root))
        from scripts.create_splits_2class import create_splits
        create_splits(data_dir)
    
    print("=" * 60)
    print("CREATING DATASETS (2 CLASSES)")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Splits directory: {splits_dir}")
    print()
    
    train_dataset = PatchDataset2Class(
        data_dir=data_dir,
        split_file=splits_dir / "train.txt",
        is_train=True
    )
    
    val_dataset = PatchDataset2Class(
        data_dir=data_dir,
        split_file=splits_dir / "val.txt",
        is_train=False
    )
    
    test_dataset = PatchDataset2Class(
        data_dir=data_dir,
        split_file=splits_dir / "test.txt",
        is_train=False
    )
    
    print()
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")
    print(f"Test dataset: {len(test_dataset)} samples")
    print()
    
    if len(train_dataset) == 0:
        print("❌ ERREUR: Dataset vide! Vérifie les fichiers.")
        print(f"   Dossier images: {data_dir / 'images'}")
        print(f"   Dossier masks: {data_dir / 'masks'}")
        return None, None, None
    
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