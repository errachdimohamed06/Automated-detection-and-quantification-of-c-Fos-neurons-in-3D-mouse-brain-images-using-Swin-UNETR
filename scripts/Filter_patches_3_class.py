#!/usr/bin/env python

import sys
from pathlib import Path
import shutil
import random
import json
import gc
from tqdm import tqdm

project_root = Path("/mnt/x/segmentation_cfos")
sys.path.insert(0, str(project_root))

from src.data.io_utils import read_tif_image, convert_image_to_array


def analyze_batch(mask_files_batch):
    """Analyse un lot de patches et retourne les catégories"""
    
    batch_neurons = []
    batch_vessels = []
    batch_background = []
    
    for msk_file in mask_files_batch:
        try:
            mask_sitk = read_tif_image(str(msk_file))
            mask = convert_image_to_array(mask_sitk)
            
            has_neuron = (mask == 1).sum() > 0
            has_vessel = (mask == 2).sum() > 0
            
            if has_neuron and has_vessel:
                batch_neurons.append(msk_file.name)
                batch_vessels.append(msk_file.name)
            elif has_neuron:
                batch_neurons.append(msk_file.name)
            elif has_vessel:
                batch_vessels.append(msk_file.name)
            else:
                batch_background.append(msk_file.name)
                
        except Exception as e:
            print(f"Erreur sur {msk_file.name}: {e}")
    
    return batch_neurons, batch_vessels, batch_background


def main():
    SAMPLE_RATE = 0.15
    SEED = 42
    BATCH_SIZE = 2000
    
    TEST_MODE = False
    TEST_SIZE = 100
    
    src_masks = Path("/mnt/x/donnees/pachs_64_3classes/masks")
    src_images = Path("/mnt/x/donnees/pachs_64_3classes/images")
    
    dst_images = Path("/mnt/x/donnees/pachs_64_3classes_filtered/images")
    dst_masks = Path("/mnt/x/donnees/pachs_64_3classes_filtered/masks")
    
    print("=" * 60)
    if TEST_MODE:
        print("MODE TEST: 30 patches uniquement")
    print("FILTRAGE DES PATCHES 3 CLASSES (BATCHED)")
    print("=" * 60)
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Seed: {SEED}")
    print()
    
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_masks.mkdir(parents=True, exist_ok=True)
    
    mask_files = sorted(src_masks.glob("*.tif"))
    
    if TEST_MODE:
        mask_files = mask_files[:TEST_SIZE]
    
    total_patches = len(mask_files)
    
    print(f"Total patches à analyser: {total_patches:,}")
    print()
    
    all_neurons = set()
    all_vessels = set()
    all_background = []
    
    num_batches = (total_patches + BATCH_SIZE - 1) // BATCH_SIZE
    
    print("Analyse par lots...")
    print("-" * 40)
    
    for batch_idx in range(0, total_patches, BATCH_SIZE):
        batch_end = min(batch_idx + BATCH_SIZE, total_patches)
        batch_files = mask_files[batch_idx:batch_end]
        
        batch_num = batch_idx // BATCH_SIZE + 1
        print(f"Lot {batch_num}/{num_batches}: {batch_idx+1} à {batch_end} ({len(batch_files)} patches)")
        
        batch_neurons, batch_vessels, batch_background = analyze_batch(batch_files)
        
        all_neurons.update(batch_neurons)
        all_vessels.update(batch_vessels)
        all_background.extend(batch_background)
        
        print(f"  Neurones: {len(all_neurons)}, Vaisseaux: {len(all_vessels)}, Fond pur: {len(all_background)}")
        
        # Libérer la mémoire
        batch_neurons = []
        batch_vessels = []
        batch_background = []
        batch_files = []
        gc.collect()
    
    print()
    print("=" * 60)
    print("STATISTIQUES")
    print("=" * 60)
    
    total_both = len(all_neurons & all_vessels)
    total_neurons_only = len(all_neurons - all_vessels)
    total_vessels_only = len(all_vessels - all_neurons)
    total_background = len(all_background)
    
    print(f"Patches avec neurones uniquement: {total_neurons_only:,}")
    print(f"Patches avec vaisseaux uniquement: {total_vessels_only:,}")
    print(f"Patches avec les deux: {total_both:,}")
    print(f"Patches fond pur: {total_background:,}")
    print("-" * 40)
    print(f"Total patches analysés: {total_patches:,}")
    print()
    
    random.seed(SEED)
    num_to_keep = int(total_background * SAMPLE_RATE)
    
    if num_to_keep == 0 and total_background > 0:
        num_to_keep = 1
    
    selected_background = random.sample(all_background, min(num_to_keep, len(all_background)))
    
    print(f"Patches fond pur à garder: {len(selected_background):,} ({SAMPLE_RATE*100:.0f}%)")
    print()
    
    final_names = set()
    final_names.update(all_neurons)
    final_names.update(all_vessels)
    final_names.update(selected_background)
    
    print(f"Total patches à copier: {len(final_names):,}")
    print()
    
    print("Copie en cours...")
    print("-" * 40)
    
    for name in tqdm(sorted(final_names), desc="Copie"):
        img_src = src_images / name
        msk_src = src_masks / name
        
        if img_src.exists() and msk_src.exists():
            shutil.copyfile(str(img_src), str(dst_images / name))
            shutil.copyfile(str(msk_src), str(dst_masks / name))
    
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
        print(f"✅ Cohérent: {len(final_images)} paires")
    else:
        print(f"⚠️ Incohérence")
    
    stats = {
        "original_patches": total_patches,
        "patches_with_neurons_only": total_neurons_only,
        "patches_with_vessels_only": total_vessels_only,
        "patches_with_both": total_both,
        "patches_pure_background": total_background,
        "sample_rate": SAMPLE_RATE,
        "background_kept": len(selected_background),
        "final_patches": len(final_images),
        "seed": SEED
    }
    
    stats_dir = project_root / "outputs" / "statistics"
    stats_dir.mkdir(parents=True, exist_ok=True)
    
    with open(stats_dir / "filter_stats_3classes.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✅ Statistiques sauvegardées: {stats_dir / 'filter_stats_3classes.json'}")
    
    if TEST_MODE:
        print("\n" + "=" * 60)
        print("✅ TEST RÉUSSI! Le script fonctionne.")
        print("Pour lancer sur toutes les données:")
        print("  - Ouvrir le script et mettre TEST_MODE = False")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("FILTRAGE TERMINÉ AVEC SUCCÈS!")
        print("=" * 60)
        print(f"Nouveau dataset: {len(final_images):,} patches")


if __name__ == "__main__":
    main()