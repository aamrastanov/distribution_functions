import numpy as np
import logging
import sys
import os
import time

# Add parent directory to sys.path to import from sibling packages
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from part1_input.image_processor import process_pipeline
from part2_build_basic_funcs.basis_generator import GaussianBasisGenerator

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
    # Paths
    # Input image is in the project root (distribution_functions)
    # parent_dir is 'img_full_restore'
    # project_root is parent of 'img_full_restore'
    project_root = os.path.dirname(parent_dir)
    image_path = os.path.join(project_root, "akvarium_in_2.png")
    
    g_inv_path = os.path.join(parent_dir, "part2_build_basic_funcs", "g_inverse_root_1000.npy")
    output_path = os.path.join(current_dir, "coefficients_1000.npy")
    
    if not os.path.exists(image_path):
        logger.error(f"Image not found: {image_path}")
        return
        
    if not os.path.exists(g_inv_path):
        logger.error(f"G inverse root matrix not found: {g_inv_path}")
        return

    # 1. Load and process image F
    logger.info(f"Processing image: {image_path}")
    try:
        F_vector = process_pipeline(image_path)
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return
        
    # F_vector is (960000, 1). Reshape to (1200, 800) for grid correlation
    F_img = F_vector.reshape(1200, 800)
    logger.info(f"Image F loaded. Shape: {F_img.shape}")
    
    # 2. Initialize Basis Generator
    bg = GaussianBasisGenerator()
    num_centers = bg.num_centers
    
    # 3. Calculate C_tilde = Phi^T * F
    logger.info("Computing C_tilde = Phi^T * F ...")
    C_tilde = np.zeros(num_centers, dtype=np.float32)
    
    denom = 2 * bg.sigma**2
    radius = int(bg.limit_sigma * bg.sigma)
    height, width = bg.height, bg.width
    
    # Precompute centers
    centers = []
    for i in range(num_centers):
        centers.append(bg.get_center_coords(i))
    centers = np.array(centers)
    
    start_time = time.time()
    
    # Optimize: Only sum over ROI
    for i in range(num_centers):
        cx, cy = centers[i]
        
        x_min = max(0, cx - radius)
        x_max = min(width, cx + radius + 1)
        y_min = max(0, cy - radius)
        y_max = min(height, cy + radius + 1)
        
        if x_min >= x_max or y_min >= y_max:
            continue
            
        # Grid for ROI
        y_idx = np.arange(y_min, y_max)
        x_idx = np.arange(x_min, x_max)
        gy, gx = np.meshgrid(y_idx, x_idx, indexing='ij')
        
        dist_sq = (gx - cx)**2 + (gy - cy)**2
        phi_roi = np.exp(-dist_sq / denom)
        
        # Extract F ROI
        F_roi = F_img[y_min:y_max, x_min:x_max]
        
        # Dot product: sum(phi_i * F)
        # Assuming discrete inner product is just element-wise sum (integration approximation)
        # Note: image_processor normalizes img to [0,1].
        val = np.sum(phi_roi * F_roi)
        
        C_tilde[i] = val
        
        if (i + 1) % 1000 == 0:
            logger.info(f"Processed {i+1}/{num_centers} centers...")
            
    logger.info(f"C_tilde calculation finished in {time.time() - start_time:.1f}s")
    
    # 4. Load G^-1/2 (9600 x 1000)
    logger.info("Loading G^-1/2 matrix...")
    M = np.load(g_inv_path)
    logger.info(f"M shape: {M.shape}")
    
    # 5. Calculate C = (G^-1/2)^T * C_tilde
    # M is (9600, 1000). M^T is (1000, 9600). C_tilde is (9600,).
    # Result C is (1000,)
    
    logger.info("Computing final coefficients C...")
    # C = np.dot(M.T, C_tilde) -> shape (1000,)
    # Or C_tilde @ M -> shape (1000,) if treated as row vector, but mathematically it's vector projection.
    # Formula: C = G^{-1/2}^T * C_tilde
    # If G^{-1/2} (M) is symmetric (it is self-adjoint), then M^T = M.
    # But shape is (9600, 1000), so it's rectangular.
    # The operator G^{-1/2} was computed as U Sigma^{-1/2}.
    # Wait, G is 9600x9600. We truncated it.
    # So M is the "rectangular pseudo-inverse root" mapping from basis space to coefficient space?
    # Yes, C = M.T @ C_tilde.
    
    C = np.dot(M.T, C_tilde)
    logger.info(f"C shape: {C.shape}")
    
    # 6. Statistics
    min_c = np.min(C)
    max_c = np.max(C)
    mean_c = np.mean(C)
    median_c = np.median(C)
    
    logger.info("=== Coefficient Stats ===")
    logger.info(f"Range: {min_c:.4f} to {max_c:.4f}")
    logger.info(f"Mean: {mean_c:.4f}")
    logger.info(f"Median: {median_c:.4f}")
    
    mid_val = (min_c + max_c) / 2
    count_lower = np.sum(C < mid_val)
    count_upper = np.sum(C >= mid_val)
    
    logger.info(f"Closer to min (< {mid_val:.4f}): {count_lower}")
    logger.info(f"Closer to max (>= {mid_val:.4f}): {count_upper}")
    
    # Save
    logger.info(f"Saving to {output_path}...")
    np.save(output_path, C)
    logger.info("Done.")

if __name__ == "__main__":
    main()
