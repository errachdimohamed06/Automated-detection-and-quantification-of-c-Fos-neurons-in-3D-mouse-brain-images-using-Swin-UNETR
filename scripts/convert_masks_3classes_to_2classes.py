#!/usr/bin/env python
"""
Convertit les masks 3 classes en 2 classes
Valeur 2 (vaisseaux) → 1 (neurones)
Version robuste avec gestion des erreurs
"""

import sys
from pathlib import Path
import shutil
import numpy as np
from tqdm import tqdm
import gc
import os

project_root = Path("/mnt/x/segmentation_cfos")
sys.path.insert(0, str(project_root))

from src.data.io_utils import read_tif_image, convert_image_to_array


def convert_mask_3to2(mask):
    """Convertit mask 3 classes (0,1,2) en 2 classes (0,1)"""
    mask_2class = mask.copy()
    mask_2class[mask_2class == 2] = 1
    return mask_2class


def safe_copy(src, dst):
    """Copie un fichier en utilisant copyfile"""
    try:
        shutil.copyfile(str(src), str(dst))
        return True
    except Exception as e:
        print(f"  Erreur copie {src.name}: {e}")
        return False


def main():
    src_images = Path("/mnt/x/donnees/pachs_64_3classes_filtered_slices/images")
    src_masks = Path("/mnt/x/donnees/pachs_64_3classes_filtered_slices/masks")

    dst_images = Path("/mnt/x/donnees/pachs_64_2classes_slices40/images")
    dst_masks = Path("/mnt/x/donnees/pachs_64_2classes_slices40/masks")

    print("=" * 60)
    print("CONVERSION 3 CLASSES → 2 CLASSES")
    print("Valeur 2 (vaisseaux) → 1 (neurones)")
    print("=" * 60)
    print()

    dst_images.mkdir(parents=True, exist_ok=True)
    dst_masks.mkdir(parents=True, exist_ok=True)
    print(f"✅ Dossiers créés: {dst_images}")
    print(f"✅ Dossiers créés: {dst_masks}")
    print()

    image_files = sorted(src_images.glob("*.tif"))
    mask_files = sorted(src_masks.glob("*.tif"))

    print(f"Total patches trouvés: {len(image_files)}")
    print()

    if len(image_files) != len(mask_files):
        print(f"⚠️ Incohérence: {len(image_files)} images vs {len(mask_files)} masks")
        return

    print("Conversion en cours...")
    print("-" * 40)

    success_count = 0
    error_count = 0

    for i, (img_file, msk_file) in enumerate(tqdm(zip(image_files, mask_files), total=len(image_files))):
        try:
            img_dst = dst_images / img_file.name
            if not safe_copy(img_file, img_dst):
                error_count += 1
                continue

            mask_sitk = read_tif_image(str(msk_file))
            mask = convert_image_to_array(mask_sitk)
            mask_2class = convert_mask_3to2(mask)

            mask_dst = dst_masks / f"{msk_file.stem}.npy"
            np.save(mask_dst, mask_2class)

            success_count += 1

        except Exception as e:
            print(f"  Erreur sur {msk_file.name}: {e}")
            error_count += 1

        if (i + 1) % 1000 == 0:
            gc.collect()

    print()
    print("=" * 60)
    print("CONVERSION TERMINÉE")
    print("=" * 60)
    print(f"✅ Patches convertis avec succès: {success_count}")
    print(f"❌ Erreurs: {error_count}")
    print()
    print(f"  - Images: {dst_images}")
    print(f"  - Masks: {dst_masks}")

    if success_count > 0:
        print()
        print("Vérification des masks convertis:")
        test_mask_path = dst_masks / f"{mask_files[0].stem}.npy"
        if test_mask_path.exists():
            test_mask = np.load(test_mask_path)
            print(f"  ✅ Fichier trouvé: {test_mask_path.name}")
            print(f"  Shape: {test_mask.shape}")
            print(f"  Valeurs uniques: {np.unique(test_mask)}")
            print(f"  Classe 0: {(test_mask == 0).sum():,} voxels")
            print(f"  Classe 1: {(test_mask == 1).sum():,} voxels")
        else:
            print(f"  ❌ Fichier non trouvé: {test_mask_path}")

    print()
    print("=" * 60)
    print("CONVERSION TERMINÉE AVEC SUCCÈS!")
    print("=" * 60)


if __name__ == "__main__":
    main()
