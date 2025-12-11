import numpy as np
import logging
import sys
import os
from scipy import sparse
from basis_generator import GaussianBasisGenerator
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class GramMatrixGenerator:
    def __init__(self, basis_generator):
        self.bg = basis_generator
        self.num_centers = self.bg.num_centers
        self.sigma = self.bg.sigma
        self.limit_sigma = self.bg.limit_sigma
        self.cutoff_dist = 6 * self.sigma
        
        # Grid parameters for intersection logic
        self.width = self.bg.width
        self.height = self.bg.height
        
    def calculate_gram_matrix(self):
        logger.info(f"Starting Gram matrix calculation. Size: {self.num_centers}x{self.num_centers}")
        logger.info(f"Sparsity cutoff distance: {self.cutoff_dist}")
        
        # Use LIL matrix for efficient construction
        G = sparse.lil_matrix((self.num_centers, self.num_centers), dtype=np.float32)
        
        start_time = time.time()
        
        # Precompute centers to avoid repeated calls
        centers = []
        for i in range(self.num_centers):
            centers.append(self.bg.get_center_coords(i))
        centers = np.array(centers) # Shape (N, 2)
        
        # Stats
        calc_count = 0
        
        # Iterate through all functions
        # This is O(N^2) but we can optimize the inner loop range because of the grid structure.
        # However, purely checking distance is simplest to implement first.
        # For 9600 items, nested loop is ~92 million checks. distance check is fast.
        
        # Optimization: use KDTree or simple grid check? 
        # Since centers are on a grid, we can just iterate neighbors.
        # But let's do the robust distance check first, optimization if needed.
        # Actually, iterating 92M times in Python is slow. 
        # But we only need to iterate j >= i.
        
        # Let's try to be smarter. 
        # The centers are ordered. 
        # We can just iterate i, and for j, we only look at indices that COULD be close.
        # But the indices mapping is a bit complex (Cy is major, Cx is minor? No, loop says:
        # cy_idx = idx // num_cx (slow), cx_idx = idx % num_cx (fast).
        # So adjacent indices are neighbors in X.
        
        denom = 2 * self.sigma**2
        radius = int(self.limit_sigma * self.sigma)
        
        for i in range(self.num_centers):
            cx_i, cy_i = centers[i]
            
            # Optimization: Pre-filter j candidates?
            # A simple loop over all j is too slow (9600*9600 iters).
            # But the matrix is sparse.
            
            # Let's rely on the grid structure of indices.
            # We want dist < 6*sigma (=90). Step=10.
            # So delta_indices in X is approx +/- 9.
            # Delta_indices in Y is approx +/- 9 rows.
            # num_cx = width / step = 800 / 10 = 80.
            
            # So valid j are in range:
            # i + k*80 + m, where k in [-9, +9], m in [-9, +9].
            
            # Determine grid coordinates of i
            grid_y_i = i // self.bg.num_cx
            grid_x_i = i % self.bg.num_cx
            
            # Range of neighbor grid coordinates
            grid_range = int(np.ceil(self.cutoff_dist / self.bg.step))
            
            min_gy = max(0, grid_y_i - grid_range)
            max_gy = min(self.bg.num_cy - 1, grid_y_i + grid_range)
            
            min_gx = max(0, grid_x_i - grid_range)
            max_gx = min(self.bg.num_cx - 1, grid_x_i + grid_range)
            
            # Iterate only through potential neighbors
            for gy in range(min_gy, max_gy + 1):
                for gx in range(min_gx, max_gx + 1):
                    j = gy * self.bg.num_cx + gx
                    
                    if j < i: continue # Symmetric, only do upper triangle
                    
                    cx_j, cy_j = centers[j]
                    
                    dist_sq = (cx_i - cx_j)**2 + (cy_i - cy_j)**2
                    
                    if dist_sq >= self.cutoff_dist**2:
                        continue
                        
                    # Calculate inner product
                    val = self.compute_inner_product(cx_i, cy_i, cx_j, cy_j, radius, denom)
                    
                    G[i, j] = val
                    G[j, i] = val # Symmetry
                    
                    calc_count += 1
            
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                logger.info(f"Processed {i+1}/{self.num_centers} rows. Calcs: {calc_count}. Time: {elapsed:.1f}s")
                
        logger.info(f"Calculation finished. Total non-zero elements: {calc_count * 2 - self.num_centers}") # approx
        
        return G.tocsr()
        
    def compute_inner_product(self, cx1, cy1, cx2, cy2, radius, denom):
        """
        Computes sum(phi_1(x,y) * phi_2(x,y)) for the overlapping region.
        """
        # Determine bounding boxes (ROI)
        # ROI for 1: [cx1-r, cx1+r]
        x1_min = max(0, cx1 - radius)
        x1_max = min(self.width, cx1 + radius + 1)
        y1_min = max(0, cy1 - radius)
        y1_max = min(self.height, cy1 + radius + 1)
        
        # ROI for 2
        x2_min = max(0, cx2 - radius)
        x2_max = min(self.width, cx2 + radius + 1)
        y2_min = max(0, cy2 - radius)
        y2_max = min(self.height, cy2 + radius + 1)
        
        # Intersection ROI
        ix_min = max(x1_min, x2_min)
        ix_max = min(x1_max, x2_max)
        iy_min = max(y1_min, y2_min)
        iy_max = min(y1_max, y2_max)
        
        if ix_min >= ix_max or iy_min >= iy_max:
            return 0.0
            
        # Grid for intersection
        # grid_y, grid_x = np.meshgrid(np.arange(iy_min, iy_max), np.arange(ix_min, ix_max), indexing='ij')
        
        # Vectorized calculation on the intersection
        # phi1 = exp(-((x-cx1)^2 + (y-cy1)^2)/denom)
        # phi2 = exp(-((x-cx2)^2 + (y-cy2)^2)/denom)
        # product = exp( - ( (x-cx1)^2 + (y-cy1)^2 + (x-cx2)^2 + (y-cy2)^2 ) / denom )
        
        # Let's perform calculation
        # We need coordinate arrays
        x_indices = np.arange(ix_min, ix_max)
        y_indices = np.arange(iy_min, iy_max)
        
        # Use meshgrid for distance calc
        gy, gx = np.meshgrid(y_indices, x_indices, indexing='ij')
        
        # Distances to center 1
        d1_sq = (gx - cx1)**2 + (gy - cy1)**2
        # Distances to center 2
        d2_sq = (gx - cx2)**2 + (gy - cy2)**2
        
        total_sq = d1_sq + d2_sq
        
        # Values
        values = np.exp(-total_sq / denom)
        
        return np.sum(values)

def main():
    output_dir = "img_full_restore/part2_build_basic_funcs"
    
    # Initialize generator to get parameters
    bg = GaussianBasisGenerator()
    
    gram_gen = GramMatrixGenerator(bg)
    
    # Calculate
    G = gram_gen.calculate_gram_matrix()
    
    # Save
    output_path = os.path.join(output_dir, "gram_matrix.npz")
    logger.info(f"Saving Gram matrix to {output_path}...")
    sparse.save_npz(output_path, G)
    logger.info("Done.")

if __name__ == "__main__":
    main()
