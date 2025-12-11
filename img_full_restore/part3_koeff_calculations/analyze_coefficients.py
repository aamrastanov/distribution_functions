import numpy as np
import logging
import sys
import os

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
    input_path = os.path.join(current_dir, "coefficients_1000.npy")
    
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        return
        
    logger.info(f"Loading coefficients from {input_path}...")
    C = np.load(input_path)
    
    # Analyze absolute values
    logger.info("Taking absolute values |C|...")
    abs_C = np.abs(C)
    
    # Basic Stats
    min_val = np.min(abs_C)
    max_val = np.max(abs_C)
    mean_val = np.mean(abs_C)
    median_val = np.median(abs_C)
    std_val = np.std(abs_C)
    
    logger.info("=== Absolute Coefficient Stats ===")
    logger.info(f"Min: {min_val:.6f}")
    logger.info(f"Max: {max_val:.6f}")
    logger.info(f"Mean: {mean_val:.6f}")
    logger.info(f"Median: {median_val:.6f}")
    logger.info(f"Std Dev: {std_val:.6f}")
    
    # Proximity Analysis
    # We define "closer to X" based on minimal distance in 1D space
    # dist_min = |x - min|
    # dist_max = |x - max|
    # dist_mean = |x - mean|
    
    d_min = np.abs(abs_C - min_val)
    d_max = np.abs(abs_C - max_val)
    d_mean = np.abs(abs_C - mean_val)
    
    # Compare distances
    # We create a stack of distances (3, N)
    d_stack = np.vstack([d_min, d_max, d_mean])
    
    # Argmin along axis 0 gives index: 0->Min, 1->Max, 2->Mean
    closest_indices = np.argmin(d_stack, axis=0)
    
    count_min = np.sum(closest_indices == 0)
    count_max = np.sum(closest_indices == 1)
    count_mean = np.sum(closest_indices == 2)
    
    logger.info("=== Proximity Analysis ===")
    logger.info("Based on shortest distance to (Min, Max, Mean):")
    logger.info(f"Closer to Min: {count_min}")
    logger.info(f"Closer to Max: {count_max}")
    logger.info(f"Closer to Mean: {count_mean}")
    
    # Additional distribution context
    logger.info("=== Percentiles ===")
    p90 = np.percentile(abs_C, 90)
    p95 = np.percentile(abs_C, 95)
    p99 = np.percentile(abs_C, 99)
    logger.info(f"90th percentile: {p90:.6f}")
    logger.info(f"99th percentile: {p99:.6f}")
    
    # Histogram
    logger.info("=== Histogram ===")
    output_plot_path = os.path.join(current_dir, "coefficients_histogram.png")
    
    import matplotlib.pyplot as plt
    
    # Text histogram (10 bins)
    hist, bins = np.histogram(abs_C, bins=10)
    for i in range(len(bins)-1):
        logger.info(f"Bin {i+1}: {bins[i]:.2f} - {bins[i+1]:.2f} -> {hist[i]}")
        
    try:
        plt.figure(figsize=(10, 6))
        plt.hist(abs_C, bins=50, color='skyblue', edgecolor='black', log=True) # Log scale y helps visualize the tail
        plt.title('Distribution of Absolute Coefficients |C| (Log Scale Y)')
        plt.xlabel('Coefficient Value')
        plt.ylabel('Count (Log Scale)')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_plot_path)
        logger.info(f"Histogram plot saved to {output_plot_path}")
    except Exception as e:
        logger.error(f"Failed to save plot: {e}")
    
    logger.info("Done.")

if __name__ == "__main__":
    main()
