#!/usr/bin/env python
"""
Script to extract patches from all images and masks
Usage: python scripts/02_create_patches.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.patching import process_all_images, get_patch_info

image_dir = project_root / "data" / "raw" / "images"
mask_dir = project_root / "data" / "raw" / "masks"
output_image_dir = project_root / "data" / "raw" / "patches_64" / "images"
output_mask_dir = project_root / "data" / "raw" / "patches_64" / "masks"

print("=" * 80)
print("PATCH EXTRACTION SCRIPT")
print("=" * 80)
print(f"Image directory: {image_dir}")
print(f"Mask directory: {mask_dir}")
print(f"Output image directory: {output_image_dir}")
print(f"Output mask directory: {output_mask_dir}")
print()

if not image_dir.exists():
    print(f"ERROR: Image directory not found: {image_dir}")
    sys.exit(1)

if not mask_dir.exists():
    print(f"ERROR: Mask directory not found: {mask_dir}")
    sys.exit(1)

info, total_patches = process_all_images(
    image_dir=image_dir,
    mask_dir=mask_dir,
    output_image_dir=output_image_dir,
    output_mask_dir=output_mask_dir,
    patch_size=64
)

print("\n" + "=" * 80)
print("VERIFICATION")
print("=" * 80)
num_image_patches = get_patch_info(output_image_dir)
num_mask_patches = get_patch_info(output_mask_dir)

if num_image_patches == num_mask_patches:
    print(f"\n✅ SUCCESS: {num_image_patches} image patches and {num_mask_patches} mask patches saved")
else:
    print(f"\n❌ ERROR: Mismatch between image patches ({num_image_patches}) and mask patches ({num_mask_patches})")