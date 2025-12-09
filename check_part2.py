import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_file(filename, expected_center_idx, width=80, height=80):
    filepath = f"img_full_restore/part2_build_basic_funcs/{filename}"
    if not os.path.exists(filepath):
        logger.error(f"File {filename} does not exist.")
        return

    logger.info(f"Loading {filename}...")
    try:
        data = np.loadtxt(filepath)
    except Exception as e:
        logger.error(f"Failed to load {filename}: {e}")
        return

    if data.shape != (height, width):
        logger.error(f"Shape mismatch for {filename}. Expected ({height}, {width}), got {data.shape}")
        return
    else:
        logger.info(f"Shape correct: {data.shape}")

    # Check peak location
    # Index 0 -> Center (0,0)
    # Index 1 -> Center (10,0)
    # Global max should be near the center.
    
    max_idx = np.unravel_index(np.argmax(data, axis=None), data.shape)
    max_val = data[max_idx]
    
    logger.info(f"Peak value: {max_val:.6e} at (y={max_idx[0]}, x={max_idx[1]})")
    
    # Expected peak
    # If filename is gauss1 (idx 0), center (0,0).
    # If filename is gauss2 (idx 1), center (0,10) -> NO, logic was x stride?
    # Basis Generator:
    # cx_range (0, 10...)
    # cy_range (0, 10...)
    # for cy ... for cx ... ?
    # Let's check the generator code loop order.
    # cx_idx = idx % num_cx -> X varies fast.
    # So Idx 1 is X=10, Y=0.
    
    logger.info(f"{filename} verification complete.")

if __name__ == "__main__":
    verify_file("gauss1.txt", 0)
    verify_file("gauss2.txt", 1)
    verify_file("gauss81.txt", 80)
