#!/usr/bin/env python3
import os
import subprocess
import itertools
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
GBR_MAIN = os.path.join(PROJECT_ROOT, "GBR", "main.py")
INPUT_IMAGE = os.path.join(BASE_DIR, "input.png")
OUTPUT_IMAGE = os.path.join(BASE_DIR, "output.png")
GT_IMAGE = os.path.join(BASE_DIR, "clear_akvarium.png")
RESULTS_FILE = os.path.join(BASE_DIR, "results_gbr_main.txt")
RESULTS_ORDERED_FILE = os.path.join(BASE_DIR, "results_gbr_main_ordered.txt")

def calculate_ssim(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        return None
    
    # Convert to grayscale if needed, or process as is
    # skimage ssim can handle multichannel if specified
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Resize if shapes don't match (should match for this task)
    if gray1.shape != gray2.shape:
        gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))
        
    score, _ = ssim(gray1, gray2, full=True)
    return score

def main():
    # Parameters
    omegas = np.round(np.arange(0.99, 1.00, 0.03), 2).tolist()
    
    # Scale pool: 1 to 5
    scale_pool = [2.0, 8.0, 9.0, 10.0]
    scale_combinations = list(itertools.combinations(scale_pool, 4))
    
    # Frequency pool: 0.1 to 2
    freq_pool = [0.5, 1.0, 2.5, 3.5, 4.0, 5.0, 6.0, 6.5, 7.5]
    freq_combinations = list(itertools.combinations(freq_pool, 5))
    
    # Generate all combinations
    all_configs = list(itertools.product(omegas, scale_combinations, freq_combinations))
    
    total_configs = len(all_configs)
    max_runs = 150
    runs_to_execute = min(total_configs, max_runs)
    
    logger.info(f"Total possible combinations: {total_configs}")
    logger.info(f"Executing max {runs_to_execute} runs.")
    
    # Results collection
    results = []
    
    # Initialize results file
    with open(RESULTS_FILE, "w") as f:
        f.write(f"Results for {os.path.basename(GBR_MAIN)}\n")
        f.write("-" * 80 + "\n")

    for i in range(runs_to_execute):
        omega, scales, freqs = all_configs[i]
        
        cmd = [
            "python3", GBR_MAIN,
            INPUT_IMAGE,
            OUTPUT_IMAGE,
            "--omega", str(omega),
            "--gabor_scales"
        ] + list(map(str, scales)) + ["--gabor_frequencies"] + list(map(str, freqs))
        
        logger.info(f"Run {i+1}/{runs_to_execute}: Omega={omega}, Scales={scales}, Freqs={freqs}")
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            
            score = calculate_ssim(OUTPUT_IMAGE, GT_IMAGE)
            if score is not None:
                res = {
                    "omega": omega,
                    "scales": scales,
                    "frequencies": freqs,
                    "ssim": score
                }
                results.append(res)
                logger.info(f"  SSIM Score: {score:.4f}")
                
                # Append to file immediately
                with open(RESULTS_FILE, "a") as f:
                    f.write(f"SSIM: {res['ssim']:.6f} | Omega: {res['omega']} | Scales: {res['scales']} | Freqs: {res['frequencies']}\n")
            else:
                logger.error(f"  Failed to calculate SSIM for run {i+1}")
                
        except subprocess.CalledProcessError as e:
            logger.error(f"  Error running GBR script: {e.stderr.decode()}")
            continue

    # Final sort and write to ordered file
    results.sort(key=lambda x: x["ssim"], reverse=True)
    with open(RESULTS_ORDERED_FILE, "w") as f:
        f.write(f"Ordered Results (Decr. SSIM) for {os.path.basename(GBR_MAIN)}\n")
        f.write("-" * 80 + "\n")
        for res in results:
            f.write(f"SSIM: {res['ssim']:.6f} | Omega: {res['omega']} | Scales: {res['scales']} | Freqs: {res['frequencies']}\n")
            
    logger.info(f"Benchmarking complete. Incremental results in {RESULTS_FILE}, sorted in {RESULTS_ORDERED_FILE}")

if __name__ == "__main__":
    main()
