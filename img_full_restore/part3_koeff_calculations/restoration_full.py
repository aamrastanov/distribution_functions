import numpy as np
import cv2
import logging
import sys
import os
import time

# Add parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

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
    coeffs_path = os.path.join(current_dir, "coefficients_1000.npy")
    g_inv_path = os.path.join(parent_dir, "part2_build_basic_funcs", "g_inverse_root_1000.npy")
    output_image_path = os.path.join(current_dir, "restored_image_full.png")
    
    # Check inputs
    for p in [coeffs_path, g_inv_path]:
        if not os.path.exists(p):
            logger.error(f"File not found: {p}")
            return

    # 1. Load Data
    logger.info("Loading data...")
    C = np.load(coeffs_path)     # (1000,)
    M = np.load(g_inv_path)      # (9600, 1000)
    
    logger.info(f"Loaded: C {C.shape}, M {M.shape}")
    
    # 2. Filtering - SKIPPED
    logger.info("Skipping filtering. Using full set of coefficients.")
    CN = C
    
    # 3. Calculate CN2 = M * CN
    logger.info("Calculating basis weights CN2 = G^-1/2 * CN ...")
    CN2 = np.dot(M, CN)
    logger.info(f"CN2 shape: {CN2.shape}")
    
    # 4. Filter CN2 Threshold (Optimization)
    threshold = 1e-6
    active_indices = np.where(np.abs(CN2) > threshold)[0]
    logger.info(f"Active CN2 weights (> {threshold}): {len(active_indices)} / {len(CN2)}")
    
    # 5. Image Reconstruction
    logger.info("Reconstructing image...")
    bg = GaussianBasisGenerator()
    width = bg.width
    height = bg.height
    
    img = np.zeros((height, width), dtype=np.float32)
    
    # Precompute constants
    denom = 2 * bg.sigma**2
    radius = int(bg.limit_sigma * bg.sigma)
    centers = [bg.get_center_coords(i) for i in range(bg.num_centers)]
    
    start_time = time.time()
    
    for i in active_indices:
        cx, cy = centers[i]
        weight = CN2[i]
        
        # Determine ROI
        x_min = max(0, cx - radius)
        x_max = min(width, cx + radius + 1)
        y_min = max(0, cy - radius)
        y_max = min(height, cy + radius + 1)
        
        if x_min >= x_max or y_min >= y_max:
            continue
            
        # Create grid for ROI
        y_idx = np.arange(y_min, y_max)
        x_idx = np.arange(x_min, x_max)
        gy, gx = np.meshgrid(y_idx, x_idx, indexing='ij')
        
        dist_sq = (gx - cx)**2 + (gy - cy)**2
        val = np.exp(-dist_sq / denom)
        
        img[y_min:y_max, x_min:x_max] += weight * val
             
    logger.info(f"Reconstruction finished in {time.time() - start_time:.1f}s")
    
    # 6. Post-processing and Saving
    min_val = np.min(img)
    max_val = np.max(img)
    logger.info(f"Reconstructed image range: {min_val:.4f} to {max_val:.4f}")
    
    img_clipped = np.clip(img, 0.0, 1.0)
    img_uint8 = (img_clipped * 255.0).astype(np.uint8)
    
    logger.info(f"Saving restored image to {output_image_path}...")
    cv2.imwrite(output_image_path, img_uint8)
    logger.info("Done.")

if __name__ == "__main__":
    main()
