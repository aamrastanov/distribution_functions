import numpy as np
import logging
import sys
import os
import time
from basis_generator import GaussianBasisGenerator
from smoothness_metric import calculate_smoothness

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
    input_matrix_path = "part2_build_basic_funcs/g_inverse_root_1000.npy"
    output_path = "part2_build_basic_funcs/modes_smoothness.npy"
    
    if not os.path.exists(input_matrix_path):
        logger.error(f"Input file not found: {input_matrix_path}")
        return

    # Load G^{-1/2} matrix (9600 x 1000)
    logger.info(f"Loading matrix from {input_matrix_path}...")
    M = np.load(input_matrix_path)
    num_centers, num_modes = M.shape
    logger.info(f"Matrix shape: {M.shape}")
    
    # Initialize generator
    bg = GaussianBasisGenerator()
    width = bg.width
    height = bg.height
    sigma = bg.sigma
    
    # Frequency limit for smoothness calculation
    # User specified: 1 / (4 * sigma)
    # Sigma is 15 -> 1/60 approx 0.016
    freq_limit = 1.0 / (4.0 * sigma)
    logger.info(f"Smoothness frequency limit: {freq_limit:.5f} (1/(4*{sigma}))")
    
    # Precompute centers
    centers = []
    for i in range(num_centers):
        centers.append(bg.get_center_coords(i))
    centers = np.array(centers)
    
    # Validation
    if num_centers != bg.num_centers:
        logger.warning(f"Matrix rows ({num_centers}) doesn't match generator centers ({bg.num_centers})!")
        
    s_values = np.zeros(num_modes, dtype=np.float32)
    
    # Batched processing
    batch_size = 50
    num_batches = int(np.ceil(num_modes / batch_size))
    
    denom = 2 * sigma**2
    # Radius for Gaussian generation (optimization)
    radius = int(bg.limit_sigma * sigma)
    
    start_total = time.time()
    
    for b in range(num_batches):
        start = b * batch_size
        end = min((b + 1) * batch_size, num_modes)
        current_batch_size = end - start
        
        logger.info(f"Processing batch {b+1}/{num_batches} (Modes {start}-{end-1})...")
        
        # Accumulator for the current batch of modes
        # Shape: (current_batch_size, height, width)
        modes_batch = np.zeros((current_batch_size, height, width), dtype=np.float32)
        
        # We need to iterate over all basis functions (9600) and add them weighted to the batch
        # This is the heavy part.
        
        # To optimize, we iterate over basis functions, generate the basis ONCE (small ROI),
        # and add it to all modes in the batch with appropriate weights.
        
        batch_weights = M[:, start:end] # Shape (9600, current_batch_size)
        
        # Optimization: Don't just iterate 9600 times blindly.
        # Check if weights are zero? M is dense, so likely not zero.
        # But we can assume M is somewhat sparse-ish if G was sparse? 
        # No, G^{-1/2} is usually dense.
        
        # Coordinate grids for ROI generation
        # We'll generate ROI grids on the fly.
        
        for i in range(num_centers):
            cx, cy = centers[i]
            
            # Get weights for this basis function across the current batch of modes
            # Shape: (current_batch_size,)
            weights = batch_weights[i, :]
            
            # If all weights are tiny, skip? (Optional optimization)
            if np.all(np.abs(weights) < 1e-9):
                continue
                
            # Generate Gaussian ROI
            x_min = max(0, cx - radius)
            x_max = min(width, cx + radius + 1)
            y_min = max(0, cy - radius)
            y_max = min(height, cy + radius + 1)
            
            if x_min >= x_max or y_min >= y_max:
                continue
            
            # Create grid for this ROI
            # grid_y, grid_x = np.meshgrid(np.arange(y_min, y_max), np.arange(x_min, x_max), indexing='ij')
            # dist_sq = (grid_x - cx)**2 + (grid_y - cy)**2
            # val = np.exp(-dist_sq / denom)
            
            # Manual grid creation to avoid overhead
            y_idx = np.arange(y_min, y_max)
            x_idx = np.arange(x_min, x_max)
            gy, gx = np.meshgrid(y_idx, x_idx, indexing='ij')
            
            dist_sq = (gx - cx)**2 + (gy - cy)**2
            basis_roi = np.exp(-dist_sq / denom) # Shape (roi_h, roi_w)
            
            # Add to accumulator
            # modes_batch[:, y_min:y_max, x_min:x_max] += weights[:, None, None] * basis_roi
            
            # Broadcasting:
            # basis_roi (H, W)
            # weights (B,) -> (B, 1, 1)
            # Result (B, H, W) added to slice
            
            weighted_roi = weights[:, np.newaxis, np.newaxis] * basis_roi
            modes_batch[:, y_min:y_max, x_min:x_max] += weighted_roi
            
        # Now we have reconstructed modes for this batch. Calculate smoothness.
        for k in range(current_batch_size):
            mode_idx = start + k
            s = calculate_smoothness(modes_batch[k], freq_limit=freq_limit)
            s_values[mode_idx] = s
            
        logger.info(f"Batch {b+1} done. Smoothness stats: Mean={np.mean(s_values[start:end]):.4f}")
        
    total_time = time.time() - start_total
    logger.info(f"All batches finished in {total_time:.1f}s.")
    
    logger.info(f"Saving S values to {output_path}...")
    np.save(output_path, s_values)
    
    # Print detailed statistics
    min_s = np.min(s_values)
    max_s = np.max(s_values)
    mean_s = np.mean(s_values)
    median_s = np.median(s_values)
    
    logger.info("=== Smoothness Stats ===")
    logger.info(f"Range: {min_s:.6f} - {max_s:.6f}")
    logger.info(f"Mean: {mean_s:.6f}")
    logger.info(f"Median: {median_s:.6f}")
    
    # Bins
    bins = [0.0, 0.9, 0.95, 0.99, 0.999, 1.0]
    hist, _ = np.histogram(s_values, bins=bins)
    
    logger.info("Distribution:")
    for i in range(len(bins)-1):
        logger.info(f"  {bins[i]:.3f} - {bins[i+1]:.3f}: {hist[i]}")
        
    # Percentiles
    p10 = np.percentile(s_values, 10)
    p90 = np.percentile(s_values, 90)
    logger.info(f"10th percentile (closer to lower): {p10:.6f}")
    logger.info(f"90th percentile (closer to upper): {p90:.6f}")
    
    logger.info("Done.")

if __name__ == "__main__":
    main()
