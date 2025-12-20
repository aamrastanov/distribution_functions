#!/usr/bin/env python3
"""
Runner script for Adaptive Bilateral Filter (ABF) and Adaptive Trilateral Filter (ATF).

Usage:
    python main.py

To switch between filters, change the FILTER_TYPE variable below.
"""

import os
import sys
import cv2
import time

# ========================================
# FILTER SELECTION - Change this to switch
# ========================================
FILTER_TYPE = "ATF"  # Options: "ABF" or "ATF"
# ========================================

# File paths (relative to this script's directory)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(SCRIPT_DIR, "block1_cahe.png")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, f"block1_cahe_{FILTER_TYPE}.png")


def main():
    # Check input file exists
    if not os.path.exists(INPUT_PATH):
        print(f"Error: Input file not found: {INPUT_PATH}")
        sys.exit(1)
    
    # Read image
    print(f"Reading image: {INPUT_PATH}")
    image = cv2.imread(INPUT_PATH)
    
    if image is None:
        print(f"Error: Could not read image: {INPUT_PATH}")
        sys.exit(1)
    
    print(f"Image shape: {image.shape}")
    
    # Select and apply filter
    start_time = time.time()
    
    if FILTER_TYPE == "ABF":
        print("Using Adaptive Bilateral Filter (ABF)")
        from abf import AdaptiveBilateralFilter
        filt = AdaptiveBilateralFilter(sigma_s=2.0, window_size=5)
    elif FILTER_TYPE == "ATF":
        print("Using Adaptive Trilateral Filter (ATF)")
        from atf import AdaptiveTrilateralFilter
        filt = AdaptiveTrilateralFilter(sigma_s=2.0, window_size=5)
    else:
        print(f"Error: Unknown filter type: {FILTER_TYPE}")
        print("Valid options: 'ABF' or 'ATF'")
        sys.exit(1)
    
    # Apply filter
    result = filt.apply(image)
    
    elapsed = time.time() - start_time
    print(f"Filtering completed in {elapsed:.2f} seconds")
    
    # Save result
    cv2.imwrite(OUTPUT_PATH, result)
    print(f"Result saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
