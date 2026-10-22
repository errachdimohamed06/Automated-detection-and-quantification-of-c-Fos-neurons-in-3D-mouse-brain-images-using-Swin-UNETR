#!/usr/bin/env python
"""
Filtre les patches par nombre de slices vides (version par lots)
Supprime les patches où plus de N slices (sur 64) ne contiennent que du fond
Traitement par lots de 2000 patches pour éviter les problèmes de mémoire
"""

import sys
from pathlib import Path
import shutil
import json
import gc
from tqdm import tqdm
import numpy as np

project_root = Path("/mnt/x/segmentation_cfos")
sys.path.insert(0, str(project_root))

from src.data.io_utils import read_tif_image, convert_image_to_array


def count_empty_slices(mask):
    """
    Compte le nombre de slices (sur l'axe Z) qui ne contiennent que du fond (classe 0)
    """
    empty_slices = 0
    for z in range(mask.shape[0]):
        slice_data = mask[z, :, :]
        if np.all(slice_data == 0):
            empty_slices += 1
    return empty_slices


def analyze_batch(mask_files, max_empty_slices, batch_idx, total_batches):
    """
    Analyse un lot de patches et retourne les patches à garder et supprimer
    """
    kept = []
    removed = []
    empty_stats = []

    for msk_file in tqdm(mask_files, desc=f"Lot {batch_idx}/{total_batches}", leave=False):
        try:
            mask_sitk = read_tif_image(str(msk_file))
            mask = convert_image_to_array(mask_sitk)

            empty_slices = count_empty_slices(mask)
            signal_slices = mask.shape[0] - empty_slices
            empty_stats.append(empty_slices)

            img_file = msk_file.parent.parent / "images" / msk_file.name

            if empty_slices > max_empty_slices:
                removed.append((img_file, msk_file, empty_slices, signal_slices))
            else:
                kept.append((img_file, msk_file, empty_slices, signal_slices))

        except Exception as e:
            print(f"Erreur sur {msk_file.name}: {e}")
            img_file = msk_file.parent.parent / "images" / msk_file.name
            kept.append((img_file, msk_file, 0, 0))

    return kept, removed, empty_stats


def main():
    MAX_EMPTY_SLICES = 40
    BATCH_SIZE = 2000
    SEED = 42

    TEST_MODE = False
    TEST_SIZE = 100

    src_images = Path("/mnt/x/donnees/pachs_64_3classes_filtered/images")
    src_masks = Path("/mnt/x/donnees/pachs_64_3classes_filtered/masks")

    dst_images = Path("/mnt/x/donnees/pachs_64_3classes_filtered_slices/images")
    dst_masks = Path("/mnt/x/donnees/pachs_64_3classes_filtered_slices/masks")

    print("=" * 60)
    if TEST_MODE:
        print("MODE TEST: 100 patches uniquement")
    print("FILTRAGE PAR NOMBRE DE SLICES VIDES (BATCHED)")
    print("=" * 60)
    print(f"Seuil: > {MAX_EMPTY_SLICES} slices vides → supprimé")
    print(f"Batch size: {BATCH_SIZE} patches")
    print(f"Seed: {SEED}")
    print()

    dst_images.mkdir(parents=True, exist_ok=True)
    dst_masks.mkdir(parents=True, exist_ok=True)

    mask_files = sorted(src_masks.glob("*.tif"))

    if TEST_MODE:
        mask_files = mask_files[:TEST_SIZE]

    total_patches = len(mask_files)

    print(f"Total patches avant filtrage: {total_patches:,}")
    print()

    num_batches = (total_patches + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Nombre de lots: {num_batches}")
    print()

    print("Analyse des patches par lots...")
    print("-" * 40)

    all_kept = []
    all_removed = []
    all_empty_stats = []

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, total_patches)
        batch_files = mask_files[start_idx:end_idx]

        print(f"\nLot {batch_idx + 1}/{num_batches}: patches {start_idx+1} à {end_idx} ({len(batch_files)} patches)")

        kept, removed, empty_stats = analyze_batch(
            batch_files,
            MAX_EMPTY_SLICES,
            batch_idx + 1,
            num_batches
        )

        all_kept.extend(kept)
        all_removed.extend(removed)
        all_empty_stats.extend(empty_stats)

        print(f"  Gardés: {len(kept)}, Supprimés: {len(removed)}")
        print(f"  Total gardés: {len(all_kept)}, Total supprimés: {len(all_removed)}")

        kept = []
        removed = []
        empty_stats = []
        batch_files = []
        gc.collect()

    print(f"\n✅ Analyse terminée!")
    print()

    print("=" * 60)
    print("STATISTIQUES")
    print("=" * 60)

    empty_stats_array = np.array(all_empty_stats)

    print(f"Patches gardés: {len(all_kept):,}")
    print(f"Patches supprimés: {len(all_removed):,}")
    print(f"Total: {total_patches:,}")
    print()
    print(f"Distribution des slices vides (sur 64):")
    print(f"  Min: {empty_stats_array.min()}")
    print(f"  Max: {empty_stats_array.max()}")
    print(f"  Moyenne: {empty_stats_array.mean():.1f}")
    print(f"  Médiane: {np.median(empty_stats_array):.1f}")
    print()

    print("Distribution détaillée:")
    ranges = [(0, 10), (11, 20), (21, 30), (31, 40), (41, 50), (51, 64)]
    for low, high in ranges:
        count = np.sum((empty_stats_array >= low) & (empty_stats_array <= high))
        print(f"  {low:2d}-{high:2d} slices vides: {count:>6,} patches")
    print()

    print("Copie des patches gardés...")
    print("-" * 40)

    for i in range(0, len(all_kept), BATCH_SIZE):
        batch_kept = all_kept[i:i+BATCH_SIZE]

        for img_file, msk_file, empty_slices, signal_slices in tqdm(
            batch_kept,
            desc=f"Copie lot {i//BATCH_SIZE + 1}",
            leave=False
        ):
            shutil.copyfile(str(img_file), str(dst_images / img_file.name))
            shutil.copyfile(str(msk_file), str(dst_masks / msk_file.name))

        batch_kept = []
        gc.collect()

    print(f"\n✅ Copie terminée!")
    print()

    final_images = sorted(dst_images.glob("*.tif"))
    final_masks = sorted(dst_masks.glob("*.tif"))

    print("=" * 60)
    print("VÉRIFICATION FINALE")
    print("=" * 60)
    print(f"Images copiées: {len(final_images):,}")
    print(f"Masks copiés: {len(final_masks):,}")

    if len(final_images) == len(final_masks):
        print(f"✅ Cohérent: {len(final_images):,} paires")
    else:
        print(f"⚠️ Incohérence")

    stats = {
        "total_patches_before": total_patches,
        "patches_kept": len(all_kept),
        "patches_removed": len(all_removed),
        "max_empty_slices": MAX_EMPTY_SLICES,
        "empty_slices_mean": float(empty_stats_array.mean()),
        "empty_slices_median": float(np.median(empty_stats_array)),
        "keep_percentage": len(all_kept) / total_patches * 100 if total_patches > 0 else 0,
        "batch_size": BATCH_SIZE,
        "test_mode": TEST_MODE
    }

    stats_dir = project_root / "outputs" / "statistics"
    stats_dir.mkdir(parents=True, exist_ok=True)

    with open(stats_dir / "slice_filter_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n✅ Statistiques sauvegardées: {stats_dir / 'slice_filter_stats.json'}")

    if TEST_MODE:
        print("\n" + "=" * 60)
        print("✅ TEST RÉUSSI! Le script fonctionne.")
        print("Pour lancer sur toutes les données:")
        print("  - Ouvrir le script et mettre TEST_MODE = False")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("FILTRAGE TERMINÉ!")
        print("=" * 60)
        print(f"Nouveau dataset: {len(final_images):,} patches")
        print(f"  - Dossier images: {dst_images}")
        print(f"  - Dossier masks: {dst_masks}")


if __name__ == "__main__":
    main()
