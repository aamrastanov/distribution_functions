import numpy as np
import logging
import sys
import os

# Add parent dir to path to import AGBR modules
sys.path.append(os.getcwd())

from AGBR.agbr_processor import AGBRProcessor

logging.basicConfig(level=logging.INFO)

def test_scaling():
    # Initialize processor
    proc = AGBRProcessor(patch_size=51, block_size=5)
    
    # Create a synthetic signal: a simple step edge in the middle of 51x51 patch
    px = np.zeros((51, 51), dtype=np.float32)
    px[25, :] = 1.0  # Just one horizontal line of gradient
    py = np.zeros((51, 51), dtype=np.float32)
    
    # Run decomposition
    cx = px @ proc.gabor_basis.T
    cy = py.T @ proc.gabor_basis.T
    
    n_filters = proc.gabor_basis.shape[0]
    offset = 51 // 2 - 5 // 2
    b_range = slice(offset, offset + 5)
    basis_b = proc.gabor_basis[:, b_range]
    
    # TEST 1: Raw reconstruction (No modification, no division by n_filters)
    # Signal -> Sum c_k * B_k
    cx_b = cx[b_range, :]
    raw_rec_x = cx_b @ basis_b
    
    # TEST 2: Reconstruction with current division
    norm_rec_x = raw_rec_x / n_filters
    
    orig_val = px[25, 25]
    raw_val = raw_rec_x[25 - offset, 25 - offset]
    norm_val = norm_rec_x[25 - offset, 25 - offset]
    
    print(f"Original center gradient value: {orig_val:.4f}")
    print(f"Raw Reconstruction sum (no div): {raw_val:.4f}")
    print(f"Current Reconstruction (div by {n_filters}): {norm_val:.4f}")
    print(f"Ratio Raw/Orig: {raw_val/orig_val:.2f}")

if __name__ == "__main__":
    test_scaling()
