#!/usr/bin/env python3
"""
GBR (Garbon Based Restore) - Main Entry Point

Reads an image, applies the GBR restoration algorithm, and saves the result.

Usage:
    python main.py [input_path] [output_path]
    
    Default: input.png -> output.png
"""

import os
import sys
import cv2
import logging
import argparse
from datetime import datetime

# Configure logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add current directory to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from gbr_processor import GBRProcessor, enhance_with_original


def main():
    """Main entry point for GBR processing."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="GBR (Garbon Based Restore) Algorithm")
    parser.add_argument("input_path", nargs="?", default="input.png", help="Path to input image")
    parser.add_argument("output_path", nargs="?", default="output.png", help="Path to output image")
    parser.add_argument("--omega", type=float, default=0.99, help="Omega parameter (default: 0.99)")
    parser.add_argument("--gabor_scales", type=float, nargs="+", default=[2.0, 8.0, 9.0, 10.0], 
                        help="Gabor scales (default: 2.0 8.0 9.0 10.0)")
    parser.add_argument("--gabor_frequencies", type=float, nargs="+", default=[0.5, 1.0, 2.5, 3.5, 6.5], 
                        help="Gabor frequencies (default: 0.5, 1.0, 2.5, 3.5, 6.5)")
    
    args = parser.parse_args()
    
    input_path = args.input_path
    if not os.path.isabs(input_path):
        input_path = os.path.join(SCRIPT_DIR, input_path)
        
    output_path = args.output_path
    if not os.path.isabs(output_path):
        output_path = os.path.join(SCRIPT_DIR, output_path)
    
    logger.info("=" * 70)
    logger.info("GBR - Garbon Based Restore Algorithm")
    logger.info("=" * 70)
    logger.info(f"Input:        {input_path}")
    logger.info(f"Output:       {output_path}")
    logger.info(f"Omega:        {args.omega}")
    logger.info(f"Gabor Scales: {args.gabor_scales}")
    logger.info(f"Gabor Freqs:  {args.gabor_frequencies}")
    
    # Check input file exists
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    # Read image
    logger.info("Reading input image...")
    image = cv2.imread(input_path)
    
    if image is None:
        logger.error(f"Failed to read image: {input_path}")
        sys.exit(1)
    
    logger.info(f"Image loaded: {image.shape[1]}x{image.shape[0]} ({image.shape[2]} channels)")
    
    # Create processor with parameters tuned for detail preservation
    processor = GBRProcessor(
        patch_size=15,
        omega=args.omega,
        guided_radius=60,
        guided_eps=1e-3,
        gabor_orientations=4,
        gabor_scales=args.gabor_scales,
        gabor_frequencies=args.gabor_frequencies,
        gabor_ksize=31,          # Gabor kernel size
        correction_epsilon=0.01,
        correction_alpha=2.0,    # REDUCED: was`` 2.0 - less high-freq suppression
        correction_beta=1.5,     # REDUCED: was 1.5 - less aggressive damping
        noise_threshold=0.01,     # DISABLED: was 0.01 - keep all details
        transmission_min=0.05,    # Minimum transmission floor
        top_percent=0.01,       # % of pixels for atmospheric light
        scale_limit=5.0         # Contrast scaling limit
    )
    
    # ========================================
    # PROCESSING MODE SELECTION
    # ========================================
    # "dehaze" - Direct dehazing with optional Gabor detail enhancement (RECOMMENDED)
    # "gradient" - Full gradient-based reconstruction (experimental)
    MODE = "gradient"
    DETAIL_BOOST = 0.3  # 0.0 to 1.0, for dehaze mode
    # ========================================
    
    # Process image
    start_time = datetime.now()
    
    if MODE == "dehaze":
        logger.info("Using DEHAZE mode (direct dehazing + optional detail enhancement)")
        result = processor.process_dehaze(image, detail_boost=DETAIL_BOOST)
    else:
        logger.info("Using GRADIENT mode (full gradient reconstruction)")
        result = processor.process(image)
    
    # Save result
    logger.info(f"Saving result to: {output_path}")
    cv2.imwrite(output_path, result)
    
    # Summary
    total_time = (datetime.now() - start_time).total_seconds()
    logger.info("")
    logger.info("=" * 70)
    logger.info("Processing Complete!")
    logger.info(f"Total time: {total_time:.2f} seconds")
    logger.info("=" * 70)
    logger.info("Output files:")
    logger.info(f"  - Restored image: {output_path}")


if __name__ == "__main__":
    main()
