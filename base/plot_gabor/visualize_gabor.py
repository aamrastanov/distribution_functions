import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Add current directory and project root to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Two levels up to reach project root from base/plot_gabor/
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from base.gabor_bank import create_gabor_bank

def visualize_gabor_bank(orientations=4, scales=[2.0, 4.0], frequencies=[0.1, 0.2], ksize=63):
    """
    Generate a grid of plots for Gabor filters in the bank.
    """
    print(f"Creating Gabor bank for visualization (ksize={ksize})...")
    bank = create_gabor_bank(
        orientations=orientations,
        scales=scales,
        frequencies=frequencies,
        ksize=ksize
    )
    
    n_filters = len(bank)
    n_cols = len(frequencies) * len(scales)
    n_rows = orientations
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    fig.suptitle(f"Gabor Filter Bank\n(Rows: Orientations, Cols: Scales x Frequencies)", fontsize=16)
    
    for i, filt in enumerate(bank):
        row = i // n_cols
        col = i % n_cols
        
        ax = axes[row, col]
        kernel = filt['kernel']
        
        # Normalize for display
        norm_kernel = cv2.normalize(kernel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        ax.imshow(norm_kernel, cmap='gray')
        ax.set_title(f"θ={np.degrees(filt['theta']):.0f}°, σ={filt['sigma']}, ω={filt['omega']}")
        ax.axis('off')
        
    plt.tight_layout()
    output_path = "gabor_bank_visualization.png"
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    # Use a smaller subset for clear visualization
    visualize_gabor_bank(
        orientations=4, 
        scales=[2.0, 5.0], 
        frequencies=[0.1, 0.3], 
        ksize=63
    )
