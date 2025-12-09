import cv2
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

def process_pipeline(filepath: str):
    """
    Executes the Part 1 processing pipeline:
    1.1 Read Image
    1.2 Grayscale
    1.3 Standardization (Resize)
    1.4 Normalization
    1.5 Vectorization
    """
    logger.info(f"Starting processing pipeline for file: {filepath}")

    # 1.1 Read Image
    logger.info("Step 1.1: Reading image...")
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        raise FileNotFoundError(f"File not found: {filepath}")
    
    img_bgr = cv2.imread(filepath)
    if img_bgr is None:
        logger.error("Failed to read image. Format might be unsupported.")
        raise ValueError("Failed to read image.")
    
    logger.info(f"Image read successfully. Shape: {img_bgr.shape}, Dtype: {img_bgr.dtype}")

    # 1.2 Grayscale
    logger.info("Step 1.2: Converting to Grayscale...")
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    logger.info(f"Converted to Grayscale. Shape: {img_gray.shape}")

    # 1.3 Standardization (Resize to 800x1200)
    target_height = 1200
    target_width = 800
    logger.info(f"Step 1.3: Standardizing size to {target_height}x{target_width}...")
    
    if img_gray.shape[0] != target_height or img_gray.shape[1] != target_width:
        logger.info(f"Original size {img_gray.shape} differs from target. Resizing using Bicubic interpolation.")
        img_resized = cv2.resize(img_gray, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
    else:
        logger.info("Image already has target dimensions. Skipping resize.")
        img_resized = img_gray

    logger.info(f"Image size after standardization: {img_resized.shape}")

    # 1.4 Normalization
    logger.info("Step 1.4: Normalizing pixel values to [0, 1]...")
    img_norm = img_resized / 255.0
    logger.info(f"Normalization complete. Min value: {img_norm.min()}, Max value: {img_norm.max()}")

    # 1.5 Vectorization
    logger.info("Step 1.5: Vectorizing to column vector...")
    img_vector = img_norm.flatten()
    # To make it a true column vector (N, 1) in numpy, strictly speaking:
    # However, user example shows F = F_norm.flatten() which results in (N,)
    # But user note says "F: 960000 * 1".
    # So I will reshape it to be explicit.
    img_vector = img_vector.reshape(-1, 1)
    
    logger.info(f"Vectorization complete. Vector shape: {img_vector.shape}")

    return img_vector

if __name__ == "__main__":
    input_file = "akvarium_in_2.png"
    try:
        result_vector = process_pipeline(input_file)
        logger.info("Pipeline completed successfully.")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)
