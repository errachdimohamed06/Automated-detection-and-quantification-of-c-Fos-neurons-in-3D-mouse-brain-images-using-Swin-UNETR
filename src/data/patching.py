import numpy as np
import patchify
import os
from pathlib import Path
from tqdm import tqdm
from .io_utils import read_tif_image, convert_image_to_array, save_array


def compute_new_size(original_size, patch_size=64):
    """
    Calculate new size according to the rule:
    - If remainder < 32: crop to lower multiple
    - If remainder >= 32: pad to upper multiple
    """
    quotient = original_size // patch_size
    remainder = original_size % patch_size
    
    if remainder == 0:
        return original_size, "no_change"
    elif remainder < patch_size // 2:
        new_size = quotient * patch_size
        return new_size, "crop"
    else:
        new_size = (quotient + 1) * patch_size
        return new_size, "pad"


def crop_3d_at_end(array, target_shape):
    """
    Crop the 3D array at the end (remove last voxels)
    """
    current_shape = array.shape
    crop_slices = []
    
    for i in range(3):
        if current_shape[i] > target_shape[i]:
            crop_slices.append(slice(0, target_shape[i]))
        else:
            crop_slices.append(slice(0, current_shape[i]))
    
    return array[tuple(crop_slices)]


def pad_3d_at_end(array, target_shape, pad_value=0):
    """
    Pad the 3D array at the end with a constant value
    """
    current_shape = array.shape
    pad_width = []
    
    for i in range(3):
        if current_shape[i] < target_shape[i]:
            pad_before = 0
            pad_after = target_shape[i] - current_shape[i]
            pad_width.append((pad_before, pad_after))
        else:
            pad_width.append((0, 0))
    
    return np.pad(array, pad_width=pad_width, mode='constant', constant_values=pad_value)


def transform_to_divisible_size(image_array, mask_array, patch_size=64):
    """
    Transform image and mask to have dimensions divisible by patch_size
    Returns: transformed image, transformed mask, and dictionary with info
    """
    original_shape = image_array.shape
    new_shape = []
    operations = []
    
    # Calculate new size for each dimension
    for dim in range(3):
        new_dim, operation = compute_new_size(original_shape[dim], patch_size)
        new_shape.append(new_dim)
        operations.append(operation)
    
    new_shape = tuple(new_shape)
    
    # Apply cropping or padding to image and mask
    transformed_image = image_array.copy()
    transformed_mask = mask_array.copy()
    
    for dim in range(3):
        if operations[dim] == "crop":
            # Crop at the end
            crop_indices = [slice(None)] * 3
            crop_indices[dim] = slice(0, new_shape[dim])
            transformed_image = transformed_image[tuple(crop_indices)]
            transformed_mask = transformed_mask[tuple(crop_indices)]
        elif operations[dim] == "pad":
            # Pad at the end
            pad_amount = new_shape[dim] - transformed_image.shape[dim]
            if pad_amount > 0:
                pad_before = [0, 0, 0]
                pad_after = [0, 0, 0]
                pad_after[dim] = pad_amount
                
                pad_width = [(pad_before[i], pad_after[i]) for i in range(3)]
                transformed_image = np.pad(transformed_image, pad_width=pad_width, mode='constant', constant_values=0)
                transformed_mask = np.pad(transformed_mask, pad_width=pad_width, mode='constant', constant_values=0)
    
    info = {
        "original_shape": original_shape,
        "new_shape": transformed_image.shape,
        "operations": operations,
        "crop_or_pad_details": {
            "Z": operations[0],
            "Y": operations[1],
            "X": operations[2]
        }
    }
    
    return transformed_image, transformed_mask, info


def extract_patches_from_pair(image, mask, patch_size=(64, 64, 64), step=None):
    """
    Extract patches from image and mask using patchify
    """
    if step is None:
        step = patch_size
    
    # Extract patches from image
    image_patches = patchify.patchify(image, patch_size, step=step)
    mask_patches = patchify.patchify(mask, patch_size, step=step)
    
    # Get number of patches in each dimension
    num_patches_z = image_patches.shape[0]
    num_patches_y = image_patches.shape[1]
    num_patches_x = image_patches.shape[2]
    
    total_patches = num_patches_z * num_patches_y * num_patches_x
    
    return image_patches, mask_patches, total_patches


def save_patches_with_counter(image_patches, mask_patches, output_image_dir, output_mask_dir, start_counter=1):
    """
    Save patches with sequential naming: patch_64_00000001.npy
    Returns the next counter value
    """
    num_patches_z = image_patches.shape[0]
    num_patches_y = image_patches.shape[1]
    num_patches_x = image_patches.shape[2]
    
    counter = start_counter
    
    for z in range(num_patches_z):
        for y in range(num_patches_y):
            for x in range(num_patches_x):
                patch_image = image_patches[z, y, x, ...]
                patch_mask = mask_patches[z, y, x, ...]
                
                # Generate sequential name
                patch_name = f"patch_64_{counter:08d}"
                
                # Save image patch
                image_path = os.path.join(output_image_dir, f"{patch_name}.npy")
                np.save(image_path, patch_image)
                
                # Save mask patch
                mask_path = os.path.join(output_mask_dir, f"{patch_name}.npy")
                np.save(mask_path, patch_mask)
                
                counter += 1
    
    return counter


def process_all_images(image_dir, mask_dir, output_image_dir, output_mask_dir, patch_size=64):
    """
    Main function to process all images and masks
    """
    # Create output directories if they don't exist
    Path(output_image_dir).mkdir(parents=True, exist_ok=True)
    Path(output_mask_dir).mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = sorted(Path(image_dir).glob("*.tif"))
    mask_files = sorted(Path(mask_dir).glob("*.tif"))
    
    print("=" * 80)
    print("PATCH EXTRACTION PIPELINE")
    print("=" * 80)
    print(f"Patch size: {patch_size}x{patch_size}x{patch_size}")
    print(f"Rule: remainder < 32 -> crop, remainder >= 32 -> pad (at the end only)")
    print(f"Total images found: {len(image_files)}")
    print(f"Total masks found: {len(mask_files)}")
    print()
    
    # Statistics
    all_info = []
    total_patches_all = 0
    global_counter = 1
    
    # Process each image-mask pair
    for i, (img_path, msk_path) in enumerate(zip(image_files, mask_files)):
        print(f"\n{'='*60}")
        print(f"Processing sample {i+1}/{len(image_files)}: {img_path.name}")
        print(f"{'='*60}")
        
        # Read image and mask
        image_sitk = read_tif_image(str(img_path))
        mask_sitk = read_tif_image(str(msk_path))
        
        # Convert to numpy arrays
        image_array = convert_image_to_array(image_sitk)
        mask_array = convert_image_to_array(mask_sitk)
        
        print(f"  Original shape (Z,Y,X): {image_array.shape}")
        print(f"  Mask shape (Z,Y,X): {mask_array.shape}")
        
        # Transform to divisible size
        transformed_image, transformed_mask, info = transform_to_divisible_size(
            image_array, mask_array, patch_size
        )
        
        print(f"  Transformed shape (Z,Y,X): {transformed_image.shape}")
        print(f"  Operations - Z: {info['operations'][0]}, Y: {info['operations'][1]}, X: {info['operations'][2]}")
        
        # Calculate voxels removed or added
        original_voxels = info['original_shape'][0] * info['original_shape'][1] * info['original_shape'][2]
        new_voxels = info['new_shape'][0] * info['new_shape'][1] * info['new_shape'][2]
        
        if new_voxels > original_voxels:
            print(f"  Added {new_voxels - original_voxels} padding voxels")
        elif new_voxels < original_voxels:
            print(f"  Removed {original_voxels - new_voxels} voxels")
        else:
            print(f"  No size change")
        
        # Extract patches
        image_patches, mask_patches, num_patches = extract_patches_from_pair(
            transformed_image, transformed_mask, patch_size=(patch_size, patch_size, patch_size)
        )
        
        print(f"  Patches per dimension (Z,Y,X): {image_patches.shape[0]} x {image_patches.shape[1]} x {image_patches.shape[2]}")
        print(f"  Total patches from this image: {num_patches}")
        
        # Save patches with sequential naming
        global_counter = save_patches_with_counter(
            image_patches, mask_patches, output_image_dir, output_mask_dir, global_counter
        )
        
        # Store info
        info["filename"] = img_path.name
        info["num_patches"] = num_patches
        all_info.append(info)
        total_patches_all += num_patches
    
    # Final summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total images processed: {len(image_files)}")
    print(f"Total patches created: {total_patches_all}")
    print(f"Output image directory: {output_image_dir}")
    print(f"Output mask directory: {output_mask_dir}")
    print()
    
    print("Details per image:")
    print("-" * 60)
    for info in all_info:
        print(f"{info['filename']:20} | Original: {info['original_shape']} -> New: {info['new_shape']} | Patches: {info['num_patches']}")
    
    print("\n" + "=" * 80)
    print("PATCH EXTRACTION COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    
    return all_info, total_patches_all


def get_patch_info(patch_dir):
    """
    Helper function to get information about saved patches
    """
    patch_files = sorted(Path(patch_dir).glob("*.npy"))
    
    print(f"Directory: {patch_dir}")
    print(f"Number of patches: {len(patch_files)}")
    
    if len(patch_files) > 0:
        first_patch = np.load(patch_files[0])
        print(f"First patch shape: {first_patch.shape}")
        print(f"First patch min/max: {first_patch.min():.3f}/{first_patch.max():.3f}")
    
    return len(patch_files)