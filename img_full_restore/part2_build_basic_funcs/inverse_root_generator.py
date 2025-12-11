import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import os
import logging
import sys
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

def main():
    input_path = "part2_build_basic_funcs/gram_matrix.npz"
    output_path = "part2_build_basic_funcs/g_inverse_root_1000.npy"
    rank = 1000
    
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        return

    logger.info(f"Loading Gram matrix from {input_path}...")
    G = scipy.sparse.load_npz(input_path)
    logger.info(f"Loaded matrix shape: {G.shape}, nnz: {G.nnz}")
    
    logger.info(f"Starting EVD (k={rank})...")
    start_time = time.time()
    
    # Calculate k largest eigenvalues/eigenvectors
    # which='LM' (Largest Magnitude) is default and usually what we want for basic functions
    vals, vecs = scipy.sparse.linalg.eigsh(G, k=rank, which='LM')
    
    elapsed = time.time() - start_time
    logger.info(f"EVD finished in {elapsed:.1f}s")
    
    # vals are eigenvalues, vecs are eigenvectors (columns)
    # They might not be sorted by magnitude, eigsh usually returns them sorted, but let's check/sort to be safe.
    # We want descending order (largest first).
    
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]
    
    logger.info(f"Top 5 eigenvalues: {vals[:5]}")
    logger.info(f"Smallest 5 eigenvalues (of {rank}): {vals[-5:]}")
    
    # Check for negative or zero eigenvalues
    if np.any(vals <= 1e-10):
        logger.warning("Found small or negative eigenvalues! This might cause stability issues.")
        # Clamp? or just warn?
        # For now, we assume G is positive semi-definite.
    
    # Calculate G^{-1/2} = U * Sigma^{-1/2}
    # Sigma is diagonal matrix of eigenvalues
    # Sigma^{-1/2} is diagonal matrix with 1/sqrt(lambda)
    
    logger.info("Computing G^{-1/2} operator...")
    
    # avoiding divide by zero just in case
    inv_sqrt_vals = 1.0 / np.sqrt(vals)
    
    # Explicit multiplication:
    # Res = U * diag(inv_sqrt_vals)
    # This is equivalent to scaling each column j of U by inv_sqrt_vals[j]
    
    # Broadcasting: (N, k) * (k,) -> each column scaled
    result = vecs * inv_sqrt_vals
    
    logger.info(f"Result shape: {result.shape}")
    
    logger.info(f"Saving to {output_path}...")
    np.save(output_path, result)
    logger.info("Done.")

if __name__ == "__main__":
    main()
