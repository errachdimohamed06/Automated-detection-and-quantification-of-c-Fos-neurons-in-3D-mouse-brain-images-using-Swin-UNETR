#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 14:04:24 2026

@author: mima
"""

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from .io_utils import read_tif_image, convert_image_to_array, show_image_info


def show_slice(image_or_array, slice_index, title="Image Slice", cmap='gray', figsize=(10, 8)):
    if isinstance(image_or_array, sitk.Image):
        array = convert_image_to_array(image_or_array)
    else:
        array = image_or_array
    
    if slice_index >= array.shape[0]:
        print(f"Slice index {slice_index} out of range. Max: {array.shape[0]-1}")
        return
    
    plt.figure(figsize=figsize)
    plt.imshow(array[slice_index, :, :], cmap=cmap)
    plt.colorbar(label='Intensity')
    plt.title(f"{title} - Slice {slice_index}")
    plt.xlabel("X (column)")
    plt.ylabel("Y (row)")
    plt.show()


def show_slices_grid(image_or_array, num_slices=4, cmap='gray', figsize=(12, 8)):
    if isinstance(image_or_array, sitk.Image):
        array = convert_image_to_array(image_or_array)
    else:
        array = image_or_array
    
    total_slices = array.shape[0]
    indices = np.linspace(0, total_slices - 1, num_slices, dtype=int)
    
    fig, axes = plt.subplots(1, num_slices, figsize=figsize)
    if num_slices == 1:
        axes = [axes]
    
    for i, idx in enumerate(indices):
        axes[i].imshow(array[idx, :, :], cmap=cmap)
        axes[i].set_title(f"Slice {idx}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def show_image_and_mask(image, mask, slice_index, figsize=(12, 5)):
    if isinstance(image, sitk.Image):
        image_array = convert_image_to_array(image)
    else:
        image_array = image
    
    if isinstance(mask, sitk.Image):
        mask_array = convert_image_to_array(mask)
    else:
        mask_array = mask
    
    if slice_index >= image_array.shape[0]:
        print(f"Slice index {slice_index} out of range. Max: {image_array.shape[0]-1}")
        return
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    
    ax1.imshow(image_array[slice_index, :, :], cmap='gray')
    ax1.set_title(f"Image - Slice {slice_index}")
    ax1.axis('off')
    
    ax2.imshow(mask_array[slice_index, :, :], cmap='gray')
    ax2.set_title(f"Mask - Slice {slice_index}")
    ax2.axis('off')
    
    overlay = np.copy(image_array[slice_index, :, :])
    overlay = (overlay - overlay.min()) / (overlay.max() - overlay.min())
    overlay = np.stack([overlay, overlay, overlay], axis=-1)
    mask_overlay = mask_array[slice_index, :, :] > 0
    overlay[mask_overlay, 0] = 1
    overlay[mask_overlay, 1] = 0
    overlay[mask_overlay, 2] = 0
    
    ax3.imshow(overlay)
    ax3.set_title(f"Overlay (Image + Mask in Red)")
    ax3.axis('off')
    
    plt.tight_layout()
    plt.show()


def get_all_images_info(image_dir, mask_dir):
    from pathlib import Path
    
    image_paths = sorted(Path(image_dir).glob("*.tif"))
    mask_paths = sorted(Path(mask_dir).glob("*.tif"))
    
    print("=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)
    print(f"Number of images: {len(image_paths)}")
    print(f"Number of masks: {len(mask_paths)}")
    print()
    
    for i, (img_path, msk_path) in enumerate(zip(image_paths, mask_paths)):
        print(f"[{i+1}] {img_path.name}")
        img = read_tif_image(str(img_path))
        msk = read_tif_image(str(msk_path))
        
        img_array = convert_image_to_array(img)
        msk_array = convert_image_to_array(msk)
        
        print(f"    Image shape (Z,Y,X): {img_array.shape}")
        print(f"    Mask shape (Z,Y,X): {msk_array.shape}")
        print(f"    Image min/max: {img_array.min():.0f} / {img_array.max():.0f}")
        print(f"    Mask unique values: {np.unique(msk_array)}")
        #print(f"    Neurons count: {(msk_array > 0).sum()}")
        print()
    
    print("=" * 60)

def explore_dataset(image_dir, mask_dir, num_samples=3):
    from pathlib import Path
    
    image_paths = sorted(Path(image_dir).glob("*.tif"))
    mask_paths = sorted(Path(mask_dir).glob("*.tif"))
    
    print(f"Total images: {len(image_paths)}")
    print(f"Total masks: {len(mask_paths)}")
    
    for i in range(min(num_samples, len(image_paths))):
        print(f"\n--- Sample {i+1}: {image_paths[i].name} ---")
        
        img = read_tif_image(str(image_paths[i]))
        msk = read_tif_image(str(mask_paths[i]))
        
        img_array = convert_image_to_array(img)
        msk_array = convert_image_to_array(msk)
        
        print(f"Image shape: {img_array.shape}")
        print(f"Mask shape: {msk_array.shape}")
        print(f"Image intensity range: [{img_array.min():.0f}, {img_array.max():.0f}]")
        print(f"Mask unique values: {np.unique(msk_array)}")
        #print(f"Number of neurons: {(msk_array > 0).sum()}")
        
        middle_slice = img_array.shape[0] // 2
        show_image_and_mask(img, msk, middle_slice, figsize=(12, 4))