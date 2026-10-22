#!/usr/bin/env python
"""
Script to filter patches into useful and unuseful folders
NO numpy aggregation functions (no np.any, np.max, np.min)
Uses simple Python loops instead
"""

import numpy as np
from pathlib import Path

project_root = Path("/mnt/x/segmentation_cfos")

def has_neurons(mask):
    """Check if mask contains at least one neuron (value 255) - NO np.any()"""
    for z in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            for x in range(mask.shape[2]):
                if mask[z, y, x] == 255:
                    return True
    return False

def has_high_intensity(image, threshold=650):
    """Check if image contains voxels above threshold - NO np.any()"""
    for z in range(image.shape[0]):
        for y in range(image.shape[1]):
            for x in range(image.shape[2]):
                if image[z, y, x] > threshold:
                    return True
    return False

def main():
    input_image_dir = project_root / "data" / "raw" / "patches_64" / "images"
    input_mask_dir = project_root / "data" / "raw" / "patches_64" / "masks"
    
    output_useful_image_dir = project_root / "data" / "raw" / "patches_64" / "useful_patches" / "images"
    output_useful_mask_dir = project_root / "data" / "raw" / "patches_64" / "useful_patches" / "masks"
    output_unuseful_image_dir = project_root / "data" / "raw" / "patches_64" / "unuseful_patches" / "images"
    output_unuseful_mask_dir = project_root / "data" / "raw" / "patches_64" / "unuseful_patches" / "masks"
    
    print("Creating directories...")
    for d in [output_useful_image_dir, output_useful_mask_dir, output_unuseful_image_dir, output_unuseful_mask_dir]:
        d.mkdir(parents=True, exist_ok=True)
        print(f"  Created: {d}")
    print()
    
    image_files = sorted(input_image_dir.glob("*.npy"))
    mask_files = sorted(input_mask_dir.glob("*.npy"))
    
    print("=" * 70)
    print("PATCH FILTERING PIPELINE")
    print("=" * 70)
    print(f"Total patches found: {len(image_files)}")
    print()
    
    stats = {
        "total": 0,
        "useful_neurons": 0,
        "useful_vessels": 0,
        "unuseful": 0
    }
    
    for i, (img_path, msk_path) in enumerate(zip(image_files, mask_files)):
        image = np.load(img_path)
        mask = np.load(msk_path)
        
        has_neuron = has_neurons(mask)
        has_high = has_high_intensity(image, 650)
        
        if has_neuron:
            classification = "useful"
            reason = "neurons"
            stats["useful_neurons"] += 1
        elif has_high:
            classification = "useful"
            reason = "vessels/artifacts"
            stats["useful_vessels"] += 1
        else:
            classification = "unuseful"
            reason = "pure background"
            stats["unuseful"] += 1
        
        if classification == "useful":
            np.save(output_useful_image_dir / img_path.name, image)
            np.save(output_useful_mask_dir / msk_path.name, mask)
        else:
            np.save(output_unuseful_image_dir / img_path.name, image)
            np.save(output_unuseful_mask_dir / msk_path.name, mask)
        
        stats["total"] += 1
        
        if (i + 1) % 500 == 0:
            print(f"Processed {i+1}/{len(image_files)} patches...")
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total patches processed: {stats['total']}")
    print()
    print("Useful patches:")
    print(f"  - Containing neurons: {stats['useful_neurons']}")
    print(f"  - Containing vessels/artifacts: {stats['useful_vessels']}")
    print(f"  - Total useful: {stats['useful_neurons'] + stats['useful_vessels']}")
    print()
    print(f"Unuseful patches (pure background): {stats['unuseful']}")
    print()
    
    useful_images = list(output_useful_image_dir.glob("*.npy"))
    useful_masks = list(output_useful_mask_dir.glob("*.npy"))
    unuseful_images = list(output_unuseful_image_dir.glob("*.npy"))
    unuseful_masks = list(output_unuseful_mask_dir.glob("*.npy"))
    
    print("VERIFICATION:")
    print(f"  Useful images: {len(useful_images)}, Useful masks: {len(useful_masks)}")
    print(f"  Unuseful images: {len(unuseful_images)}, Unuseful masks: {len(unuseful_masks)}")
    
    if len(useful_images) == len(useful_masks) and len(unuseful_images) == len(unuseful_masks):
        print("  ✅ All pairs are consistent!")
    else:
        print("  ❌ WARNING: Mismatch detected!")
    
    total_useful = len(useful_images)
    total_unuseful = len(unuseful_images)
    total = total_useful + total_unuseful
    
    print()
    print("FINAL STATISTICS:")
    print(f"  Useful patches for training: {total_useful} ({total_useful/total*100:.1f}%)")
    print(f"  Discarded pure background: {total_unuseful} ({total_unuseful/total*100:.1f}%)")
    print("=" * 70)

if __name__ == "__main__":
    print("Starting patch filtering...")
    print()
    main()
    print("\nDone!")