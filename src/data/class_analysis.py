"""
class_analysis.py
Functions to analyze class distribution in masks (0 = background, 255 = neurons)
Masks are not converted yet (original values: 0 and 255)
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def load_all_masks(mask_dir, max_masks=None):
    """
    Load all mask .npy files from a directory
    
    Parameters:
        mask_dir: path to directory containing .npy mask files
        max_masks: maximum number of masks to load (for testing)
    
    Returns:
        list of numpy arrays, list of filenames
    """
    mask_paths = sorted(Path(mask_dir).glob("*.npy"))
    
    if max_masks:
        mask_paths = mask_paths[:max_masks]
    
    masks = []
    filenames = []
    
    for path in mask_paths:
        mask = np.load(path)
        masks.append(mask)
        filenames.append(path.name)
    
    return masks, filenames


def compute_voxel_class_statistics(masks):
    """
    Compute voxel-level class distribution statistics
    
    Parameters:
        masks: list of mask arrays with values 0 and 255
    
    Returns:
        dict with statistics
    """
    total_voxels = 0
    total_class_0 = 0  # background (0)
    total_class_255 = 0  # neurons (255)
    
    class_0_per_mask = []
    class_255_per_mask = []
    ratio_per_mask = []
    
    for mask in masks:
        class_0 = np.sum(mask == 0)
        class_255 = np.sum(mask == 255)
        total = class_0 + class_255
        
        total_voxels += total
        total_class_0 += class_0
        total_class_255 += class_255
        
        class_0_per_mask.append(class_0)
        class_255_per_mask.append(class_255)
        
        if class_255 > 0:
            ratio_per_mask.append(class_0 / class_255)
        else:
            ratio_per_mask.append(float('inf'))
    
    stats = {
        "total_voxels": total_voxels,
        "total_class_0": total_class_0,
        "total_class_255": total_class_255,
        "class_0_percentage": (total_class_0 / total_voxels) * 100,
        "class_255_percentage": (total_class_255 / total_voxels) * 100,
        "class_ratio": total_class_0 / max(total_class_255, 1),
        "class_0_per_mask": class_0_per_mask,
        "class_255_per_mask": class_255_per_mask,
        "ratio_per_mask": ratio_per_mask,
        "num_masks": len(masks),
        "masks_with_neurons": sum(1 for c in class_255_per_mask if c > 0),
        "masks_without_neurons": sum(1 for c in class_255_per_mask if c == 0)
    }
    
    return stats


def find_extreme_masks(masks, filenames, class_255_per_mask):
    """
    Find masks with extreme values
    """
    class_255_array = np.array(class_255_per_mask)
    
    results = {
        "max_neurons_voxels": {
            "index": np.argmax(class_255_array),
            "filename": filenames[np.argmax(class_255_array)],
            "voxel_count": np.max(class_255_array)
        },
        "min_neurons_voxels_nonzero": {
            "index": np.argmin(class_255_array[class_255_array > 0]) if np.any(class_255_array > 0) else None,
            "filename": filenames[np.argmin(class_255_array[class_255_array > 0])] if np.any(class_255_array > 0) else None,
            "voxel_count": np.min(class_255_array[class_255_array > 0]) if np.any(class_255_array > 0) else 0
        },
        "zero_neurons": {
            "indices": np.where(class_255_array == 0)[0].tolist(),
            "filenames": [filenames[i] for i in np.where(class_255_array == 0)[0]],
            "count": np.sum(class_255_array == 0)
        }
    }
    
    return results


def print_statistics(stats):
    """
    Print voxel-level class distribution statistics
    """
    print("=" * 70)
    print("VOXEL CLASS DISTRIBUTION ANALYSIS")
    print("=" * 70)
    print()
    print(f"Number of patches analyzed: {stats['num_masks']}")
    print(f"Patches containing neurons: {stats['masks_with_neurons']}")
    print(f"Patches without neurons (pure background): {stats['masks_without_neurons']}")
    print()
    print("Total voxels across all patches:")
    print(f"  Total voxels: {stats['total_voxels']:,}")
    print(f"  Class 0 (background, bruit, artefacts): {stats['total_class_0']:,} ({stats['class_0_percentage']:.4f}%)")
    print(f"  Class 255 (neurones): {stats['total_class_255']:,} ({stats['class_255_percentage']:.6f}%)")
    print()
    print(f"Class ratio (Class 0 / Class 255): 1 : {stats['class_ratio']:.2f}")
    print()
    print("Per patch statistics (voxels class 255 = neurones):")
    print(f"  Min voxels per patch: {np.min(stats['class_255_per_mask']):,}")
    print(f"  Max voxels per patch: {np.max(stats['class_255_per_mask']):,}")
    print(f"  Mean voxels per patch: {np.mean(stats['class_255_per_mask']):.2f}")
    print(f"  Median voxels per patch: {np.median(stats['class_255_per_mask']):.2f}")
    print(f"  Std voxels per patch: {np.std(stats['class_255_per_mask']):.2f}")
    print()
    print("=" * 70)


def plot_class_distribution(stats, save_path=None):
    """
    Create visualizations of voxel class distribution
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Pie chart of class distribution
    ax1 = axes[0, 0]
    sizes = [stats['total_class_0'], stats['total_class_255']]
    labels = [f'Class 0 (Background)\n{stats["class_0_percentage"]:.4f}%', 
              f'Class 255 (Neurons)\n{stats["class_255_percentage"]:.6f}%']
    colors = ['#333333', '#FF4444']
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.4f%%', startangle=90)
    ax1.set_title('Voxel Class Distribution (All Patches)')
    
    # Plot 2: Histogram of class 255 voxels per patch
    ax2 = axes[0, 1]
    class_255_nonzero = [c for c in stats['class_255_per_mask'] if c > 0]
    if len(class_255_nonzero) > 0:
        ax2.hist(class_255_nonzero, bins=50, edgecolor='black', alpha=0.7, color='red')
        ax2.set_xlabel('Number of voxels in Class 255 (neurons) per patch')
        ax2.set_ylabel('Number of patches')
        ax2.set_title(f'Distribution of Neuron Voxels per Patch\n(mean: {np.mean(class_255_nonzero):.1f}, median: {np.median(class_255_nonzero):.1f})')
        ax2.axvline(np.mean(class_255_nonzero), color='blue', linestyle='--', label=f'Mean: {np.mean(class_255_nonzero):.1f}')
        ax2.axvline(np.median(class_255_nonzero), color='green', linestyle='--', label=f'Median: {np.median(class_255_nonzero):.1f}')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'No neuron voxels in any patch', ha='center', va='center')
        ax2.set_title('No neurons found')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Log scale histogram of class 255 voxels per patch
    ax3 = axes[1, 0]
    if len(class_255_nonzero) > 0:
        ax3.hist(class_255_nonzero, bins=50, edgecolor='black', alpha=0.7, color='red', log=True)
        ax3.set_xlabel('Number of voxels in Class 255 (neurons) per patch')
        ax3.set_ylabel('Number of patches (log scale)')
        ax3.set_title('Distribution of Neuron Voxels per Patch (Log Scale)')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No neuron voxels in any patch', ha='center', va='center')
        ax3.set_title('No neurons found')
    
    # Plot 4: Box plot of class 255 voxels per patch
    ax4 = axes[1, 1]
    if len(class_255_nonzero) > 0:
        bp = ax4.boxplot([class_255_nonzero], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightcoral')
        ax4.set_ylabel('Number of neuron voxels per patch')
        ax4.set_title(f'Box Plot of Neuron Voxels per Patch\n(Median: {np.median(class_255_nonzero):.1f})')
        ax4.grid(True, alpha=0.3, axis='y')
    else:
        ax4.text(0.5, 0.5, 'No neuron voxels in any patch', ha='center', va='center')
        ax4.set_title('No neurons found')
    
    plt.suptitle('Voxel Class Distribution Analysis (Masks with values 0 and 255)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_class_distribution_per_patch(stats, num_bins=50):
    """
    Plot distribution of Class 0 and Class 255 separately
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Class 0 distribution
    ax1 = axes[0]
    ax1.hist(stats['class_0_per_mask'], bins=num_bins, edgecolor='black', alpha=0.7, color='gray')
    ax1.set_xlabel('Number of Class 0 voxels per patch (background)')
    ax1.set_ylabel('Number of patches')
    ax1.set_title(f'Distribution of Background Voxels (Class 0)')
    ax1.grid(True, alpha=0.3)
    
    # Class 255 distribution (only non-zero)
    ax2 = axes[1]
    class_255_nonzero = [c for c in stats['class_255_per_mask'] if c > 0]
    if len(class_255_nonzero) > 0:
        ax2.hist(class_255_nonzero, bins=num_bins, edgecolor='black', alpha=0.7, color='red')
        ax2.set_xlabel('Number of Class 255 voxels per patch (neurons)')
        ax2.set_ylabel('Number of patches')
        ax2.set_title('Distribution of Neuron Voxels (Class 255)')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No neuron voxels found', ha='center', va='center')
        ax2.set_title('No neurons detected')
    
    plt.suptitle('Class Distribution per Patch (Separate Views)', fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_extreme_masks(masks, filenames, extreme_info, num_display=4):
    """
    Visualize masks with extreme values
    """
    # Display mask with most neuron voxels
    idx_max = extreme_info["max_neurons_voxels"]["index"]
    mask_max = masks[idx_max]
    filename_max = extreme_info["max_neurons_voxels"]["filename"]
    count_max = extreme_info["max_neurons_voxels"]["voxel_count"]
    
    # Display a mask with zero neurons if exists
    zero_indices = extreme_info["zero_neurons"]["indices"]
    zero_filenames = extreme_info["zero_neurons"]["filenames"]
    
    num_plots = 1 + (1 if len(zero_indices) > 0 else 0)
    num_plots = min(num_plots, num_display)
    
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5))
    if num_plots == 1:
        axes = [axes]
    
    # Plot mask with most neuron voxels
    middle_z = mask_max.shape[0] // 2
    axes[0].imshow(mask_max[middle_z, :, :], cmap='gray')
    axes[0].set_title(f'Most Neuron Voxels\n{filename_max}\n{count_max:,} voxels Class 255')
    axes[0].axis('off')
    
    # Plot a mask with zero neurons
    if len(zero_indices) > 0 and num_plots > 1:
        idx_zero = zero_indices[0]
        mask_zero = masks[idx_zero]
        filename_zero = zero_filenames[0]
        middle_z = mask_zero.shape[0] // 2
        axes[1].imshow(mask_zero[middle_z, :, :], cmap='gray')
        axes[1].set_title(f'Zero Neurons\n{filename_zero}\nNo Class 255 voxels')
        axes[1].axis('off')
    
    plt.suptitle('Extreme Cases: Most Neuron Voxels vs Zero Neurons', fontsize=14)
    plt.tight_layout()
    plt.show()


def generate_report(stats, output_path=None):
    """
    Generate a text report of the analysis
    """
    report = []
    report.append("=" * 70)
    report.append("VOXEL CLASS DISTRIBUTION ANALYSIS REPORT")
    report.append("=" * 70)
    report.append("")
    report.append(f"Number of patches analyzed: {stats['num_masks']}")
    report.append(f"Patches containing neurons: {stats['masks_with_neurons']}")
    report.append(f"Patches without neurons (pure background): {stats['masks_without_neurons']}")
    report.append("")
    report.append("Total voxels across all patches:")
    report.append(f"  Total voxels: {stats['total_voxels']:,}")
    report.append(f"  Class 0 (background, bruit, artefacts): {stats['total_class_0']:,} ({stats['class_0_percentage']:.4f}%)")
    report.append(f"  Class 255 (neurones): {stats['total_class_255']:,} ({stats['class_255_percentage']:.6f}%)")
    report.append("")
    report.append(f"Class ratio (Class 0 / Class 255): 1 : {stats['class_ratio']:.2f}")
    report.append("")
    report.append("Per patch statistics (voxels Class 255):")
    report.append(f"  Min voxels per patch: {np.min(stats['class_255_per_mask']):,}")
    report.append(f"  Max voxels per patch: {np.max(stats['class_255_per_mask']):,}")
    report.append(f"  Mean voxels per patch: {np.mean(stats['class_255_per_mask']):.2f}")
    report.append(f"  Median voxels per patch: {np.median(stats['class_255_per_mask']):.2f}")
    report.append("")
    
    # Add recommendation based on class imbalance
    report.append("RECOMMENDATION FOR LOSS FUNCTION:")
    if stats['class_ratio'] > 10000:
        report.append("  EXTREME IMBALANCE: Use Dice Loss or Tversky Loss")
        report.append("  Consider Focal Loss with gamma=2 or higher")
    elif stats['class_ratio'] > 1000:
        report.append("  SEVERE IMBALANCE: Use Dice Loss or Focal Loss (gamma=2)")
    elif stats['class_ratio'] > 100:
        report.append("  STRONG IMBALANCE: Use Dice Loss or Weighted CrossEntropy")
    elif stats['class_ratio'] > 10:
        report.append("  MODERATE IMBALANCE: Weighted CrossEntropy or Dice Loss")
    else:
        report.append("  BALANCED: Standard CrossEntropy may work")
    
    report.append("")
    report.append("=" * 70)
    
    report_text = "\n".join(report)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_text)
    
    print(report_text)
    
    return report_text