import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    # Paths
    # 1. Coefficients
    coef_path = os.path.join(current_dir, "coefficients_1000.npy")
    # 2. Smoothness (in part2_build_basic_funcs)
    smooth_path = os.path.join(parent_dir, "part2_build_basic_funcs", "modes_smoothness.npy")
    
    # Output path
    output_path = os.path.join(current_dir, "combined_analysis.png")
    
    if not os.path.exists(coef_path):
        logger.error(f"Coefficients file not found: {coef_path}")
        return
        
    if not os.path.exists(smooth_path):
        logger.error(f"Smoothness file not found: {smooth_path}")
        return
        
    logger.info("Loading data...")
    C = np.load(coef_path)
    S = np.load(smooth_path)
    
    # Validation
    if C.shape != S.shape:
        logger.warning(f"Shapes mismatch: C {C.shape}, S {S.shape}")
        
    num_modes = min(len(C), len(S))
    indices = np.arange(num_modes)
    
    abs_C = np.abs(C[:num_modes])
    S = S[:num_modes]
    
    logger.info(f"Plotting for {num_modes} modes...")
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot 1: Absolute Coefficients
    # Using log scale for y because we know it's heavy tailed
    ax1.bar(indices, abs_C, color='royalblue', width=1.0, alpha=0.7)
    # Also plot regular plot line?
    # ax1.plot(indices, abs_C, color='darkblue', linewidth=0.5)
    ax1.set_ylabel('Absolute Coefficient |C|', fontsize=12)
    ax1.set_title('Coefficients Magnitude vs Mode Index', fontsize=14)
    ax1.grid(True, alpha=0.3)
    # Optional: Log scale
    # ax1.set_yscale('log') 
    
    # Plot 2: Smoothness
    ax2.plot(indices, S, color='forestgreen', linewidth=1.0)
    ax2.fill_between(indices, S, color='lightgreen', alpha=0.3)
    ax2.set_ylabel('Smoothness S', fontsize=12)
    ax2.set_xlabel('Mode Index', fontsize=12)
    ax2.set_title('Mode Smoothness vs Mode Index', fontsize=14)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    logger.info(f"Combined plot saved to {output_path}")

if __name__ == "__main__":
    main()
