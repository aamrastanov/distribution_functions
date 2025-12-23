import os
import sys
import cv2
import logging
import argparse
import time

# Add project root and current directoy for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from agbr_processor import AGBRProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="AGBR (Optimized Garbon Based Restore) Algorithm")
    
    # Paths
    parser.add_argument("input_path", nargs="?", default="input.png", help="Path to input image")
    parser.add_argument("output_path", nargs="?", default="output.png", help="Path to output image")
    
    # AGBR Parameters
    parser.add_argument("--patch_size", type=int, default=51, help="Window size (default: 51)")
    parser.add_argument("--stride", type=int, default=5, help="Stride step (default: 5)")
    parser.add_argument("--block_size", type=int, default=5, help="Middle block size (default: 5)")
    parser.add_argument("--omega", type=float, default=0.99, help="Omega parameter (default: 0.99)")
    parser.add_argument("--beta_high", type=float, default=0.6, help="Beta High (default: 0.7)")
    parser.add_argument("--beta_low", type=float, default=0.4, help="Beta Low (default: 0.3)")
    parser.add_argument("--r_min", type=float, default=0.05, help="R min threshold (default: 0.05)")
    parser.add_argument("--eps_thresh", type=float, default=1e-9, help="Energy epsilon threshold (default: 1e-5)")
    
    # Gabor Parameters
    parser.add_argument("--gabor_scales", type=float, nargs="+", default=[1.0, 2.0, 4.0, 5.0, 7.0, 8.0], 
                        help="Gabor scales (default: 1.0 2.0 4.0 5.0 7.0 8.0)")
    parser.add_argument("--gabor_frequencies", type=float, nargs="+", default=[0.5, 2.0, 5.0, 10.0, 17.0, 23.0], 
                        help="Gabor frequencies (default: 0.5 2.0 5.0 10.0 17.0 23.0)")
    
    args = parser.parse_args()
    
    # Resolve paths
    input_path = args.input_path
    if not os.path.exists(input_path) and not os.path.isabs(input_path):
        input_path = os.path.join(SCRIPT_DIR, input_path)
        
    output_path = args.output_path
    if not os.path.isabs(output_path):
        # For output, we might want it in SCRIPT_DIR if not specified otherwise
        # but let's be consistent and check if directory exists
        if not os.path.dirname(output_path) or not os.path.exists(os.path.dirname(output_path)):
             output_path = os.path.join(SCRIPT_DIR, output_path)
        
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        return

    logger.info("=" * 70)
    logger.info("AGBR - Optimized Garbon Based Restore Algorithm")
    logger.info("=" * 70)
    logger.info(f"Input:         {input_path}")
    logger.info(f"Output:        {output_path}")
    logger.info(f"Patch Size (k): {args.patch_size}")
    logger.info(f"Stride (S):    {args.stride}")
    logger.info(f"Omega:         {args.omega}")
    logger.info(f"Beta High:     {args.beta_high}")
    logger.info(f"Beta Low:      {args.beta_low}")
    logger.info(f"R Min:         {args.r_min}")
    logger.info(f"Gabor Scales:  {args.gabor_scales}")
    logger.info(f"Gabor Freqs:   {args.gabor_frequencies}")
    
    # Load image
    image = cv2.imread(input_path)
    if image is None:
        logger.error("Failed to load image!")
        return

    # Initialize Processor
    processor = AGBRProcessor(
        patch_size=args.patch_size,
        stride=args.stride,
        block_size=args.block_size,
        omega=args.omega,
        beta_high=args.beta_high,
        beta_low=args.beta_low,
        r_min=args.r_min,
        eps_thresh=args.eps_thresh,
        gabor_scales=args.gabor_scales,
        gabor_frequencies=args.gabor_frequencies
    )

    # Process
    try:
        output_image = processor.process(image)
        
        # Save result
        cv2.imwrite(output_path, output_image)
        logger.info(f"Saved restored image to: {output_path}")
        logger.info("AGBR Processing Complete!")
    except Exception as e:
        logger.exception(f"An error occurred during processing: {e}")

if __name__ == "__main__":
    main()
