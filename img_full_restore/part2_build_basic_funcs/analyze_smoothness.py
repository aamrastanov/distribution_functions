import numpy as np
import logging
import sys
import os
import matplotlib.pyplot as plt

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
    input_path = os.path.join(current_dir, "modes_smoothness.npy")
    output_plot_path = os.path.join(current_dir, "smoothness_histogram.png")
    
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        return
        
    logger.info(f"Loading smoothness values from {input_path}...")
    S = np.load(input_path)
    
    # S values are already positive (0 to 1). abs() not needed but harmless.
    vals = S
    
    # Basic Stats
    min_val = np.min(vals)
    max_val = np.max(vals)
    mean_val = np.mean(vals)
    median_val = np.median(vals)
    std_val = np.std(vals)
    
    logger.info("=== Smoothness Stats ===")
    logger.info(f"Min: {min_val:.6f}")
    logger.info(f"Max: {max_val:.6f}")
    logger.info(f"Mean: {mean_val:.6f}")
    logger.info(f"Median: {median_val:.6f}")
    logger.info(f"Std Dev: {std_val:.6f}")
    
    # Proximity Analysis
    d_min = np.abs(vals - min_val)
    d_max = np.abs(vals - max_val)
    d_mean = np.abs(vals - mean_val)
    
    # Compare distances
    d_stack = np.vstack([d_min, d_max, d_mean])
    closest_indices = np.argmin(d_stack, axis=0)
    
    count_min = np.sum(closest_indices == 0)
    count_max = np.sum(closest_indices == 1)
    count_mean = np.sum(closest_indices == 2)
    
    logger.info("=== Proximity Analysis ===")
    logger.info("Based on shortest distance to (Min, Max, Mean):")
    logger.info(f"Closer to Min: {count_min}")
    logger.info(f"Closer to Max: {count_max}")
    logger.info(f"Closer to Mean: {count_mean}")
    
    # Percentiles
    logger.info("=== Percentiles ===")
    p10 = np.percentile(vals, 10)
    p25 = np.percentile(vals, 25)
    p75 = np.percentile(vals, 75)
    p90 = np.percentile(vals, 90)
    logger.info(f"10th percentile: {p10:.6f}")
    logger.info(f"25th percentile: {p25:.6f}")
    logger.info(f"75th percentile: {p75:.6f}")
    logger.info(f"90th percentile: {p90:.6f}")

    # Histogram
    logger.info("=== Histogram ===")
    
    # Text histogram (10 bins)
    hist, bins = np.histogram(vals, bins=10, range=(0.0, 1.0)) # Fix range 0-1 for clarity
    for i in range(len(bins)-1):
        logger.info(f"Bin {i+1}: {bins[i]:.2f} - {bins[i+1]:.2f} -> {hist[i]}")
        
    try:
        plt.figure(figsize=(10, 6))
        # Use more bins for the plot
        plt.hist(vals, bins=50, color='lightgreen', edgecolor='black', range=(0.0, 1.0))
        plt.title('Distribution of Mode Smoothness S')
        plt.xlabel('Smoothness Value')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_plot_path)
        logger.info(f"Histogram plot saved to {output_plot_path}")
    except Exception as e:
        logger.error(f"Failed to save plot: {e}")
    
    logger.info("Done.")

if __name__ == "__main__":
    main()
