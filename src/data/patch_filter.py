"""
patch_filter.py
Filter patches based on mask values and image intensity
Useful patches: contain neurons (mask=255) OR high intensity voxels (vessels/artifacts)
Unuseful patches: pure background with low intensity
"""

import numpy as np
from pathlib import Path
import shutil


def load_patch(file_path):
    """Load a .npy patch file and return numpy array"""
    return np.load(file_path)


def has_neurons(mask):
    """Check if mask contains at least one neuron (value 255)"""
    # Direct check without using np.any
    for val in np.unique(mask):
        if val == 255:
            return True
    return False


def has_high_intensity(image, threshold=650):
    """Check if image contains voxels above threshold"""
    # Direct check without using np.any
    if image.max() > threshold:
        return True
    return False


def classify_patch(image, mask, intensity_threshold=650):
    """
    Classify a patch as useful or unuseful
    
    Returns:
        ("useful" or "unuseful", reason)
    """
    # Case 1: Mask has neurons (value 255)
    if has_neurons(mask):
        return "useful", "neurons"
    
    # Case 2: Mask has only zeros, check image intensity
    if has_high_intensity(image, intensity_threshold):
        return "useful", "vessels/artifacts"
    
    # Case 3: Pure background with low intensity
    return "unuseful", "pure background"


def filter_all_patches(
    input_image_dir,
    input_mask_dir,
    output_useful_image_dir,
    output_useful_mask_dir,
    output_unuseful_image_dir,
    output_unuseful_mask_dir,
    intensity_threshold=650
):
    """
    Filter all patches and separate into useful and unuseful folders
    """
    # Create output directories
    output_useful_image_dir.mkdir(parents=True, exist_ok=True)
    output_useful_mask_dir.mkdir(parents=True, exist_ok=True)
    output_unuseful_image_dir.mkdir(parents=True, exist_ok=True)
    output_unuseful_mask_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all patch files
    image_files = sorted(input_image_dir.glob("*.npy"))
    mask_files = sorted(input_mask_dir.glob("*.npy"))
    
    # Verify matching files
    if len(image_files) != len(mask_files):
        print(f"WARNING: {len(image_files)} images vs {len(mask_files)} masks")
        min_len = min(len(image_files), len(mask_files))
        image_files = image_files[:min_len]
        mask_files = mask_files[:min_len]
    
    # Statistics
    stats = {
        "total": len(image_files),
        "useful_neurons": 0,
        "useful_vessels": 0,
        "unuseful_pure_background": 0,
        "total_useful": 0,
        "total_unuseful": 0,
        "errors": 0
    }
    
    print("=" * 70)
    print("PATCH FILTERING PIPELINE")
    print("=" * 70)
    print(f"Intensity threshold: {intensity_threshold}")
    print(f"Total patches found: {stats['total']}")
    print()
    
    for i, (img_path, msk_path) in enumerate(zip(image_files, mask_files)):
        try:
            # Load image and mask
            image = load_patch(img_path)
            mask = load_patch(msk_path)
            
            # Check if shapes match
            if image.shape != mask.shape:
                print(f"WARNING: Shape mismatch for {img_path.name}: image {image.shape} vs mask {mask.shape}")
                stats["errors"] += 1
                continue
            
            # Classify
            classification, reason = classify_patch(image, mask, intensity_threshold)
            
            # Update statistics and copy files
            if classification == "useful":
                if reason == "neurons":
                    stats["useful_neurons"] += 1
                else:
                    stats["useful_vessels"] += 1
                stats["total_useful"] += 1
                
                shutil.copy2(img_path, output_useful_image_dir / img_path.name)
                shutil.copy2(msk_path, output_useful_mask_dir / msk_path.name)
            else:
                stats["unuseful_pure_background"] += 1
                stats["total_unuseful"] += 1
                
                shutil.copy2(img_path, output_unuseful_image_dir / img_path.name)
                shutil.copy2(msk_path, output_unuseful_mask_dir / msk_path.name)
            
            # Print progress every 500 patches
            if (i + 1) % 500 == 0:
                print(f"Processed {i+1}/{stats['total']} patches...")
                
        except Exception as e:
            print(f"ERROR processing {img_path.name}: {e}")
            stats["errors"] += 1
    
    # Print summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total patches processed: {stats['total']}")
    print(f"Errors encountered: {stats['errors']}")
    print()
    print("Useful patches:")
    print(f"  - Containing neurons: {stats['useful_neurons']}")
    print(f"  - Containing vessels/artifacts: {stats['useful_vessels']}")
    print(f"  - Total useful: {stats['total_useful']}")
    print()
    print("Unuseful patches:")
    print(f"  - Pure background: {stats['unuseful_pure_background']}")
    print(f"  - Total unuseful: {stats['total_unuseful']}")
    print()
    if stats['total'] > 0:
        print(f"Useful percentage: {stats['total_useful']/stats['total']*100:.2f}%")
        print(f"Unuseful percentage: {stats['total_unuseful']/stats['total']*100:.2f}%")
    print()
    print("Output directories:")
    print(f"  Useful images: {output_useful_image_dir}")
    print(f"  Useful masks: {output_useful_mask_dir}")
    print(f"  Unuseful images: {output_unuseful_image_dir}")
    print(f"  Unuseful masks: {output_unuseful_mask_dir}")
    print("=" * 70)
    
    return stats


def get_filter_statistics(
    useful_image_dir,
    useful_mask_dir,
    unuseful_image_dir,
    unuseful_mask_dir
):
    """
    Get statistics about the filtered datasets
    """
    useful_images = list(useful_image_dir.glob("*.npy"))
    useful_masks = list(useful_mask_dir.glob("*.npy"))
    unuseful_images = list(unuseful_image_dir.glob("*.npy"))
    unuseful_masks = list(unuseful_mask_dir.glob("*.npy"))
    
    print("=" * 50)
    print("FILTERED DATASET STATISTICS")
    print("=" * 50)
    print(f"Useful patches - images: {len(useful_images)}")
    print(f"Useful patches - masks: {len(useful_masks)}")
    print(f"Unuseful patches - images: {len(unuseful_images)}")
    print(f"Unuseful patches - masks: {len(unuseful_masks)}")
    
    if len(useful_images) == len(useful_masks):
        print(f"✅ Useful dataset consistent: {len(useful_images)} pairs")
    else:
        print(f"❌ WARNING: Mismatch in useful dataset")
    
    if len(unuseful_images) == len(unuseful_masks):
        print(f"✅ Unuseful dataset consistent: {len(unuseful_images)} pairs")
    else:
        print(f"❌ WARNING: Mismatch in unuseful dataset")
    
    return {
        "useful_pairs": len(useful_images),
        "unuseful_pairs": len(unuseful_images)
    }


def diagnose_patch(file_path):
    """
    Diagnostic function to inspect a single patch without triggering numpy errors
    """
    data = np.load(file_path)
    print(f"File: {file_path}")
    print(f"  Type: {type(data)}")
    print(f"  Shape: {data.shape}")
    print(f"  Dtype: {data.dtype}")
    
    # Use Python's min/max to avoid numpy issues
    flat_data = data.flatten()
    min_val = flat_data.min()
    max_val = flat_data.max()
    print(f"  Min value: {min_val}")
    print(f"  Max value: {max_val}")
    
    # Get unique values by iterating (slow but safe)
    unique_vals = set()
    for val in flat_data[:100000]:  # Check first 100k voxels
        unique_vals.add(val)
    
    unique_list = sorted(list(unique_vals))
    print(f"  Number of unique values (sample): {len(unique_list)}")
    if len(unique_list) <= 10:
        print(f"  Unique values: {unique_list}")
    else:
        print(f"  First 10 unique values: {unique_list[:10]}")
    
    return data