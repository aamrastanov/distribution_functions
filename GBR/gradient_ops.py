"""
Gradient Operations module for GBR algorithm.
Step 2: Gradient computation
Step 6: Poisson integration
"""

import numpy as np
import cv2
from scipy.fft import fft2, ifft2
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def compute_gradients(image: np.ndarray) -> tuple:
    """
    Compute horizontal and vertical gradients using forward difference.
    gx[i, j] = image[i, j+1] - image[i, j]
    gy[i, j] = image[i+1, j] - image[i, j]
    
    Args:
        image: Input image (grayscale or BGR)
    
    Returns:
        Tuple (gx, gy) - gradient maps
    """
    logger.info("Computing image gradients using forward difference")
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
    else:
        gray = image.astype(np.float64)
    
    # Forward difference grad
    gx = np.roll(gray, -1, axis=1) - gray
    gy = np.roll(gray, -1, axis=0) - gray
    
    logger.info(f"Gradients computed. gx range: [{gx.min():.2f}, {gx.max():.2f}], "
                f"gy range: [{gy.min():.2f}, {gy.max():.2f}]")
    
    return gx, gy


def compute_gradients_color(image: np.ndarray) -> tuple:
    """
    Compute gradients for color image (per channel) using forward difference.
    
    Args:
        image: BGR image
    
    Returns:
        Tuple (gx, gy) where each is (H, W, 3)
    """
    logger.info("Computing color image gradients (forward difference)")
    
    image_float = image.astype(np.float64)
    
    gx = np.roll(image_float, -1, axis=1) - image_float
    gy = np.roll(image_float, -1, axis=0) - image_float
    
    logger.info("Color gradients computed for all 3 channels")
    return gx, gy


def compute_divergence(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """
    Compute divergence using backward difference to match forward gradient.
    div[i, j] = (gx[i, j] - gx[i, j-1]) + (gy[i, j] - gy[i-1, j])
    
    Args:
        gx: Horizontal gradient (forward diff)
        gy: Vertical gradient (forward diff)
    
    Returns:
        Divergence field
    """
    # Backward difference div
    div_x = gx - np.roll(gx, 1, axis=gx.ndim-2) # Correct for both 2D and 3D arrays
    div_y = gy - np.roll(gy, 1, axis=0) if gx.ndim == 2 else gy - np.roll(gy, 1, axis=0)
    
    # For safety with axes in 3D (H, W, 3)
    if gx.ndim == 3:
        div_x = gx - np.roll(gx, 1, axis=1)
        div_y = gy - np.roll(gy, 1, axis=0)
    else:
        div_x = gx - np.roll(gx, 1, axis=1)
        div_y = gy - np.roll(gy, 1, axis=0)
        
    divergence = div_x + div_y
    
    return divergence


def poisson_solve(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """
    Solve Poisson equation: Δf = div(g) using FFT.
    
    Reconstructs image from gradient field.
    
    Args:
        gx: Horizontal gradient
        gy: Vertical gradient
    
    Returns:
        Reconstructed image
    """
    logger.info("Solving Poisson equation using FFT")
    
    h, w = gx.shape[:2]
    
    # Handle color images
    if len(gx.shape) == 3:
        result = np.zeros_like(gx)
        for c in range(3):
            result[:, :, c] = _poisson_solve_channel(gx[:, :, c], gy[:, :, c])
        return result
    else:
        return _poisson_solve_channel(gx, gy)


def _poisson_solve_channel(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """
    Solve Poisson equation for single channel.
    """
    h, w = gx.shape
    
    # Compute divergence
    div = compute_divergence(gx, gy)
    
    # Create frequency domain coordinates
    u = np.fft.fftfreq(h)
    v = np.fft.fftfreq(w)
    U, V = np.meshgrid(v, u)
    
    # Laplacian in frequency domain
    # Δ -> -4π²(u² + v²)
    denom = (np.sin(np.pi * U) ** 2 + np.sin(np.pi * V) ** 2)
    denom = np.maximum(denom, 1e-10)  # Avoid division by zero
    
    # FFT of divergence
    div_fft = fft2(div)
    
    # Solve in frequency domain (using the spectral Laplacian for forward/backward diff)
    f_fft = div_fft / (-4 * denom)
    f_fft[0, 0] = 0  # Set DC component to zero
    
    # Inverse FFT
    f = np.real(ifft2(f_fft))
    
    logger.info(f"Poisson solution computed. Range: [{f.min():.2f}, {f.max():.2f}]")
    return f


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to [0, 255] range.
    """
    min_val = image.min()
    max_val = image.max()
    
    if max_val - min_val < 1e-10:
        return np.zeros_like(image, dtype=np.uint8)
    
    normalized = (image - min_val) / (max_val - min_val) * 255.0
    return np.clip(normalized, 0, 255).astype(np.uint8)
