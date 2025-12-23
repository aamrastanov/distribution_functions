"""
Gabor Filter Bank module for GBR algorithm.
Step 3: Gabor decomposition
Step 5: Gradient reconstruction
"""

import numpy as np
import cv2
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def create_gabor_kernel(ksize: int, sigma: float, theta: float, 
                        lambd: float, gamma: float = 0.5, psi: float = 0) -> np.ndarray:
    """
    Create a single Gabor kernel.
    
    Args:
        ksize: Kernel size
        sigma: Standard deviation of Gaussian envelope
        theta: Orientation in radians
        lambd: Wavelength of sinusoidal factor (lambda)
        gamma: Spatial aspect ratio
        psi: Phase offset
    
    Returns:
        Gabor kernel (ksize x ksize)
    """
    kernel = cv2.getGaborKernel(
        (ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_64F
    )
    return kernel


def create_gabor_bank(orientations: int = 8, 
                      scales: list = None,
                      frequencies: list = None,
                      ksize: int = 31) -> list:
    """
    Create a bank of Gabor filters with various orientations, scales, and frequencies.
    
    Args:
        orientations: Number of orientation angles
        scales: List of sigma values (Gaussian envelope scales)
        frequencies: List of spatial frequencies (1/lambda)
        ksize: Kernel size (default 31)
    
    Returns:
        List of dicts with 'kernel', 'theta', 'sigma', 'omega' keys
    """
    logger.info(f"Creating Gabor filter bank: {orientations} orientations, ksize={ksize}")
    
    if scales is None:
        scales = [2.0, 4.0, 8.0]  # sigma values
    if frequencies is None:
        frequencies = [0.1, 0.2, 0.3]  # 1/lambda
    
    bank = []
    
    for theta_idx in range(orientations):
        theta = theta_idx * np.pi / orientations
        
        for sigma in scales:
            for freq in frequencies:
                lambd = 1.0 / freq  # wavelength
                
                kernel = create_gabor_kernel(ksize, sigma, theta, lambd)
                
                bank.append({
                    'kernel': kernel,
                    'theta': theta,
                    'sigma': sigma,
                    'omega': freq,  # frequency
                    'lambd': lambd
                })
    
    logger.info(f"Gabor bank created with {len(bank)} filters")
    logger.info(f"  - Scales (sigma): {scales}")
    logger.info(f"  - Frequencies (omega): {frequencies}")
    
    return bank


def gabor_decompose(gx: np.ndarray, gy: np.ndarray, bank: list) -> list:
    """
    Decompose gradient fields using Gabor filter bank.
    
    Performs convolution of gradient fields with each Gabor filter
    to extract coefficients representing different structural modes.
    
    Args:
        gx: Horizontal gradient
        gy: Vertical gradient
        bank: Gabor filter bank
    
    Returns:
        List of coefficient dicts with 'cx', 'cy', 'theta', 'sigma', 'omega'
    """
    logger.info(f"Decomposing gradients with {len(bank)} Gabor filters")
    
    coefficients = []
    
    for i, filt in enumerate(bank):
        kernel = filt['kernel'].astype(np.float64)
        
        # Use cv2.filter2D for fast convolution (much faster than scipy)
        cx = cv2.filter2D(gx, cv2.CV_64F, kernel)
        cy = cv2.filter2D(gy, cv2.CV_64F, kernel)
        
        coefficients.append({
            'cx': cx,
            'cy': cy,
            'theta': filt['theta'],
            'sigma': filt['sigma'],
            'omega': filt['omega'],
            'kernel': kernel
        })
        
        if (i + 1) % 20 == 0:
            logger.info(f"  Processed {i+1}/{len(bank)} filters")
    
    logger.info("Gabor decomposition complete")
    return coefficients


def gabor_decompose_color(gx: np.ndarray, gy: np.ndarray, bank: list) -> list:
    """
    Decompose color gradient fields.
    
    Args:
        gx: Horizontal gradient (H, W, 3)
        gy: Vertical gradient (H, W, 3)
        bank: Gabor filter bank
    
    Returns:
        List of coefficient dicts for each channel
    """
    logger.info("Decomposing color gradients with Gabor bank")
    
    coefficients = []
    
    for c in range(3):
        logger.info(f"Processing channel {c}")
        channel_coeffs = gabor_decompose(gx[:, :, c], gy[:, :, c], bank)
        coefficients.append(channel_coeffs)
    
    return coefficients


def gabor_reconstruct(coefficients: list) -> tuple:
    """
    Reconstruct gradient fields from Gabor coefficients.
    
    Sum all corrected coefficient-weighted basis functions.
    
    Args:
        coefficients: List of coefficient dicts
    
    Returns:
        Tuple (gx_hat, gy_hat) - reconstructed gradients
    """
    logger.info(f"Reconstructing gradients from {len(coefficients)} Gabor components")
    
    # Initialize with first coefficient shape
    shape = coefficients[0]['cx'].shape
    gx_hat = np.zeros(shape, dtype=np.float64)
    gy_hat = np.zeros(shape, dtype=np.float64)
    
    for coeff in coefficients:
        # The coefficient is already the result of convolution
        # For reconstruction, we sum contributions
        gx_hat += coeff['cx']
        gy_hat += coeff['cy']
    
    # Normalize by number of filters
    n_filters = len(coefficients)
    gx_hat /= n_filters
    gy_hat /= n_filters
    
    logger.info(f"Gradients reconstructed. gx range: [{gx_hat.min():.2f}, {gx_hat.max():.2f}]")
    return gx_hat, gy_hat


def gabor_reconstruct_color(coefficients: list) -> tuple:
    """
    Reconstruct color gradient fields.
    
    Args:
        coefficients: List of 3 channel coefficient lists
    
    Returns:
        Tuple (gx_hat, gy_hat) with shape (H, W, 3)
    """
    logger.info("Reconstructing color gradients")
    
    results = []
    for c in range(3):
        gx_c, gy_c = gabor_reconstruct(coefficients[c])
        results.append((gx_c, gy_c))
    
    h, w = results[0][0].shape
    gx_hat = np.zeros((h, w, 3), dtype=np.float64)
    gy_hat = np.zeros((h, w, 3), dtype=np.float64)
    
    for c in range(3):
        gx_hat[:, :, c] = results[c][0]
        gy_hat[:, :, c] = results[c][1]
    
    return gx_hat, gy_hat


def create_gabor_1d_basis(ksize: int, scales: list, frequencies: list) -> np.ndarray:
    """
    Create a 1D Gabor basis matrix for windowed processing.
    Each row is a normalized 1D Gabor function.
    
    Args:
        ksize: Length of the 1D Gabor filter (match window size)
        scales: List of sigma values
        frequencies: List of frequencies (cycles per window)
    
    Returns:
        Basis matrix (N_filters x ksize)
    """
    logger.info(f"Creating 1D Gabor basis: ksize={ksize}, {len(scales)} scales, {len(frequencies)} freqs")
    
    basis = []
    
    for sigma in scales:
        for freq in frequencies:
            # Frequency to wavelength conversion: cycles per window -> pixels per cycle (lambda)
            lambd = ksize / freq
            
            # Using cv2.getGaborKernel for 1D: width=ksize, height=1
            # theta=0 for horizontal, psi=0 for Cosine, psi=pi/2 for Sine
            g_cos = cv2.getGaborKernel((ksize, 1), sigma, 0, lambd, 1.0, 0, ktype=cv2.CV_64F).flatten()
            g_sin = cv2.getGaborKernel((ksize, 1), sigma, 0, lambd, 1.0, np.pi/2, ktype=cv2.CV_64F).flatten()
            
            # L2 Normalization
            norm_cos = np.linalg.norm(g_cos)
            norm_sin = np.linalg.norm(g_sin)
            
            if norm_cos > 1e-10:
                basis.append(g_cos / norm_cos)
            if norm_sin > 1e-10:
                basis.append(g_sin / norm_sin)
                
    return np.array(basis, dtype=np.float32)
