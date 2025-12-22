"""
Dark Channel Prior module for GBR algorithm.
Step 1: Estimation of atmospheric parameters.

Based on: "Single Image Haze Removal Using Dark Channel Prior" (He et al.)
"""

import numpy as np
import cv2
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def compute_dark_channel(image: np.ndarray, patch_size: int = 15) -> np.ndarray:
    """
    Compute the dark channel of an image.
    
    For each pixel, find the minimum value among all color channels
    within a local patch.
    
    Args:
        image: Input BGR image (H, W, 3), values in [0, 255]
        patch_size: Size of local patch (default 15x15)
    
    Returns:
        Dark channel map (H, W), values in [0, 255]
    """
    logger.info(f"Computing dark channel with patch size {patch_size}x{patch_size}")
    
    # Normalize to [0, 1]
    img_norm = image.astype(np.float64) / 255.0
    
    # Take minimum across color channels
    min_channel = np.min(img_norm, axis=2)
    
    # Apply minimum filter with patch
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark_channel = cv2.erode(min_channel, kernel)
    
    logger.info(f"Dark channel computed. Range: [{dark_channel.min():.3f}, {dark_channel.max():.3f}]")
    return dark_channel


def estimate_atmospheric_light(image: np.ndarray, dark_channel: np.ndarray, 
                              top_percent: float = 0.001) -> np.ndarray:
    """
    Estimate atmospheric light A from the brightest pixels in dark channel.
    
    Args:
        image: Input BGR image
        dark_channel: Dark channel map
        top_percent: Percentage of brightest pixels to consider (default 0.1%)
    
    Returns:
        Atmospheric light vector [Ab, Ag, Ar]
    """
    logger.info(f"Estimating atmospheric light using top {top_percent*100:.3f}% pixels")
    
    h, w = dark_channel.shape
    num_pixels = int(h * w * top_percent)
    num_pixels = max(num_pixels, 1)  # At least 1 pixel
    
    # Flatten and find brightest pixels in dark channel
    dark_flat = dark_channel.ravel()
    indices = np.argsort(dark_flat)[-num_pixels:]
    
    # Get corresponding pixels from original image
    img_flat = image.reshape(-1, 3)
    brightest_pixels = img_flat[indices]
    
    # Average to get atmospheric light
    A = np.mean(brightest_pixels, axis=0)
    
    logger.info(f"Atmospheric light estimated: A = [{A[0]:.1f}, {A[1]:.1f}, {A[2]:.1f}]")
    return A


def estimate_transmission(image: np.ndarray, A: np.ndarray, 
                          omega: float = 0.95, patch_size: int = 15,
                          transmission_min: float = 0.1) -> np.ndarray:
    """
    Estimate transmission map t(x).
    
    t(x) = 1 - omega * min_{y in Omega(x)} ( min_{c in RGB} ( I_c(y) / A_c ) )
    
    Args:
        image: Input BGR image
        A: Atmospheric light vector
        omega: Haze preservation coefficient (default 0.95)
        patch_size: Local patch size
        transmission_min: Minimum transmission value (default 0.1)
    
    Returns:
        Transmission map (H, W), values in [0.1, 1]
    """
    logger.info(f"Estimating transmission map with omega={omega}")
    
    # Normalize image by atmospheric light
    img_norm = image.astype(np.float64) / A.reshape(1, 1, 3)
    
    # Clamp to avoid extreme values
    img_norm = np.clip(img_norm, 0, 1)
    
    # Compute dark channel of normalized image
    min_channel = np.min(img_norm, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark_norm = cv2.erode(min_channel, kernel)
    
    # Transmission estimate
    transmission = 1.0 - omega * dark_norm
    
    # Clamp to valid range
    transmission = np.clip(transmission, transmission_min, 1.0)
    
    logger.info(f"Transmission map computed. Range: [{transmission.min():.3f}, {transmission.max():.3f}]")
    return transmission


def guided_filter(I: np.ndarray, p: np.ndarray, 
                  radius: int = 60, eps: float = 1e-3) -> np.ndarray:
    """
    Apply guided filter for edge-preserving smoothing.
    
    Used to refine the transmission map to align with image edges.
    
    Args:
        I: Guidance image (grayscale, [0, 1])
        p: Input to filter (transmission map)
        radius: Filter radius
        eps: Regularization parameter
    
    Returns:
        Filtered output
    """
    logger.info(f"Applying guided filter with radius={radius}, eps={eps}")
    
    # Box filter helper
    def box_filter(img, r):
        return cv2.boxFilter(img, -1, (2*r+1, 2*r+1))
    
    I = I.astype(np.float64)
    p = p.astype(np.float64)
    
    mean_I = box_filter(I, radius)
    mean_p = box_filter(p, radius)
    mean_Ip = box_filter(I * p, radius)
    mean_II = box_filter(I * I, radius)
    
    cov_Ip = mean_Ip - mean_I * mean_p
    var_I = mean_II - mean_I * mean_I
    
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    
    mean_a = box_filter(a, radius)
    mean_b = box_filter(b, radius)
    
    q = mean_a * I + mean_b
    
    logger.info("Guided filter applied successfully")
    return q


def refine_transmission(image: np.ndarray, transmission: np.ndarray,
                        radius: int = 60, eps: float = 1e-3,
                        transmission_min: float = 0.1) -> np.ndarray:
    """
    Refine transmission map using guided filter.
    
    Args:
        image: Original BGR image
        transmission: Raw transmission map
        radius: Guided filter radius
        eps: Regularization
        transmission_min: Minimum transmission value
    
    Returns:
        Refined transmission map
    """
    logger.info("Refining transmission map with guided filter")
    
    # Use grayscale image as guidance
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    
    refined = guided_filter(gray, transmission, radius, eps)
    refined = np.clip(refined, transmission_min, 1.0)
    
    logger.info(f"Transmission refined. Range: [{refined.min():.3f}, {refined.max():.3f}]")
    return refined
