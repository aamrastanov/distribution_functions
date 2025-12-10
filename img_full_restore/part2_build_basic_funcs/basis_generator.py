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

class GaussianBasisGenerator:
    def __init__(self, width=800, height=1200, sigma=15, step=10, limit_sigma=3):
        self.width = width
        self.height = height
        self.sigma = sigma
        self.step = step
        self.limit_sigma = limit_sigma
        
        # Ranges
        self.cx_range = np.arange(0, width, step) # 0, 10, ...
        self.cy_range = np.arange(0, height, step) # 0, 10, ...
        
        self.num_cx = len(self.cx_range)
        self.num_cy = len(self.cy_range)
        self.num_centers = self.num_cx * self.num_cy
        
        logger.info(f"Basis Generator initialized. Grid: {height}x{width}, Centers: {self.num_centers}")

    def get_center_coords(self, idx):
        """
        Returns (cx, cy) for a given global index.
        Index order: Y changes slowly, X changes fast?
        Wait, I need to match the user's indices exactly.
        User: 
          Idx 0: (0, 0)
          Idx 1: (10, 0) -> X changes.
          Idx 79: (790, 0) -> X changes.
          Idx 80: (0, 10) -> Y changed.
        So:
          cy_index = idx // num_cx
          cx_index = idx % num_cx of the range
        """
        if idx < 0 or idx >= self.num_centers:
            raise ValueError(f"Index {idx} out of range [0, {self.num_centers-1}]")
            
        cy_idx = idx // self.num_cx
        cx_idx = idx % self.num_cx
        
        cx = self.cx_range[cx_idx]
        cy = self.cy_range[cy_idx]
        return cx, cy

    def generate_column(self, idx):
        """
        Generates the dense column vector (reshaped to 2D) for the basis function at idx.
        Returns: numpy array of shape (height, width)
        """
        cx, cy = self.get_center_coords(idx)
        # logger.info(f"Generating basis for index {idx} at center ({cx}, {cy})")
        
        # Optimization: Only compute within radius, leave rest as zeros
        # But for 'savetxt' we need the full 1200x800 array.
        # We can create a zeros array and fill the ROI.
        
        img = np.zeros((self.height, self.width), dtype=np.float32)
        
        radius = int(self.limit_sigma * self.sigma)
        
        x_min = max(0, cx - radius)
        x_max = min(self.width, cx + radius + 1)
        y_min = max(0, cy - radius)
        y_max = min(self.height, cy + radius + 1)
        
        grid_y, grid_x = np.meshgrid(np.arange(y_min, y_max), np.arange(x_min, x_max), indexing='ij')
        
        dist_sq = (grid_x - cx)**2 + (grid_y - cy)**2
        denom = 2 * self.sigma**2
        
        val = np.exp(-dist_sq / denom)
        
        img[y_min:y_max, x_min:x_max] = val
        
        return img

    def save_column_to_file(self, idx, output_dir):
        filename = f"gauss{idx + 1}.txt"
        filepath = os.path.join(output_dir, filename)
        logger.info(f"Saving {filename} (Index {idx})...")
        
        img = self.generate_column(idx)
        
        # Crop: take top-left 80x80 pixels
        # img shape is (1200, 800).
        # We want rows 0-79 and cols 0-79.
        img_cropped = img[:80, :80]
        
        logger.info(f"Cropped shape: {img_cropped.shape}")
        
        # Save as text matrix
        np.savetxt(filepath, img_cropped, fmt='%.6e', delimiter=' ')

def main():
    output_dir = "img_full_restore/part2_build_basic_funcs"
    os.makedirs(output_dir, exist_ok=True)
    
    generator = GaussianBasisGenerator()
    
    # Validation of indices
    # Test index 0
    cx, cy = generator.get_center_coords(0)
    logger.info(f"Check Idx 0: ({cx}, {cy}) -> Expect (0, 0)")
    
    # Test index 1
    cx, cy = generator.get_center_coords(1)
    logger.info(f"Check Idx 1: ({cx}, {cy}) -> Expect (10, 0)")

    # Test index 80 (81st item)
    # 800 width / 10 step = 80 items per row.
    # So index 80 should be start of next row: (0, 10).
    cx, cy = generator.get_center_coords(80)
    logger.info(f"Check Idx 80: ({cx}, {cy}) -> Expect (0, 10)")
    
    indices_to_save = [
        0, 1,           # gauss1, gauss2
        79, 80, 81,     # gauss80, gauss81, gauss82
        159, 160, 161   # gauss160, gauss161, gauss162
    ]
    
    for idx in indices_to_save:
        if idx < generator.num_centers:
            generator.save_column_to_file(idx, output_dir)
        else:
            logger.warning(f"Index {idx} out of bounds, skipping.")
            
    logger.info("Part 2 completed successfully.")

if __name__ == "__main__":
    main()
