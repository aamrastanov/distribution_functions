"""
Coefficient Correction module for GBR algorithm.
Step 4: Physical correction of Gabor coefficients.

Applies frequency-dependent amplification based on transmission map.
"""

import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def compute_weight_function(omega: float, sigma: float, 
                            alpha: float = 1.0, beta: float = 0.5) -> float:
    """
    Compute frequency-dependent weight W(omega, sigma).
    
    This weight limits amplification of high frequencies to prevent noise explosion.
    
    Formula: W = 1 / (1 + alpha * omega^beta)
    
    Args:
        omega: Spatial frequency
        sigma: Scale parameter
        alpha: Amplification control
        beta: Frequency exponent
    
    Returns:
        Weight value in (0, 1]
    """
    # Higher frequency = lower weight = less amplification
    weight = 1.0 / (1.0 + alpha * (omega ** beta))
    return weight


def correct_coefficients(coefficients: list, transmission: np.ndarray,
                         epsilon: float = 0.1, 
                         alpha: float = 2.0, beta: float = 1.5,
                         noise_threshold: float = 0.01) -> list:
    """
    Apply physical correction to Gabor coefficients.
    
    Formula: c̃_k = c_k * (1 / max(t(x), ε)) * W(ω_k, σ_k)
    
    Where:
        - t(x) is transmission map
        - W is frequency-dependent weight limiting high-frequency amplification
        - ε prevents division by zero
    
    Args:
        coefficients: List of Gabor coefficient dicts
        transmission: Transmission map (H, W)
        epsilon: Minimum transmission value
        alpha: Weight function parameter
        beta: Weight function parameter
        noise_threshold: Threshold for noise suppression
    
    Returns:
        Corrected coefficients
    """
    logger.info(f"Correcting {len(coefficients)} coefficient sets")
    logger.info(f"  Parameters: epsilon={epsilon}, alpha={alpha}, beta={beta}")
    
    # Ensure transmission has minimum value
    t_safe = np.maximum(transmission, epsilon)
    
    # Amplification factor based on transmission
    amplification = 1.0 / t_safe
    
    logger.info(f"  Amplification range: [{amplification.min():.2f}, {amplification.max():.2f}]")
    
    corrected = []
    
    for i, coeff in enumerate(coefficients):
        omega = coeff['omega']
        sigma = coeff['sigma']
        
        # Frequency-dependent weight
        W = compute_weight_function(omega, sigma, alpha, beta)
        
        # Apply correction
        cx_corrected = coeff['cx'] * amplification * W
        cy_corrected = coeff['cy'] * amplification * W
        
        # Threshold filtering (noise suppression)
        cx_corrected = threshold_filter(cx_corrected, noise_threshold)
        cy_corrected = threshold_filter(cy_corrected, noise_threshold)
        
        corrected.append({
            'cx': cx_corrected,
            'cy': cy_corrected,
            'theta': coeff['theta'],
            'sigma': sigma,
            'omega': omega,
            'kernel': coeff['kernel'],
            'weight_applied': W
        })
    
    logger.info("Coefficient correction complete")
    return corrected


def correct_coefficients_color(coefficients: list, transmission: np.ndarray,
                               **kwargs) -> list:
    """
    Apply correction to color (3-channel) coefficients.
    
    Args:
        coefficients: List of 3 channel coefficient lists
        transmission: Transmission map
        **kwargs: Additional parameters for correct_coefficients
    
    Returns:
        List of corrected coefficients for each channel
    """
    logger.info("Correcting color coefficients (3 channels)")
    
    corrected = []
    for c in range(3):
        logger.info(f"Correcting channel {c}")
        channel_corrected = correct_coefficients(coefficients[c], transmission, **kwargs)
        corrected.append(channel_corrected)
    
    return corrected


def threshold_filter(coeffs: np.ndarray, threshold: float) -> np.ndarray:
    """
    Apply soft thresholding to suppress noise.
    
    Coefficients with magnitude below threshold * max are attenuated.
    
    Args:
        coeffs: Coefficient array
        threshold: Threshold as fraction of max value
    
    Returns:
        Filtered coefficients
    """
    max_val = np.abs(coeffs).max()
    if max_val < 1e-10:
        return coeffs
    
    thresh_abs = threshold * max_val
    
    # Soft thresholding
    sign = np.sign(coeffs)
    magnitude = np.abs(coeffs)
    
    # Attenuate small coefficients
    filtered = sign * np.maximum(0, magnitude - thresh_abs)
    
    return filtered


def adaptive_correction(coefficients: list, transmission: np.ndarray,
                        local_window: int = 15) -> list:
    """
    Apply locally adaptive correction based on local transmission statistics.
    
    In regions with very low transmission (t < 0.2), apply less aggressive
    amplification to prevent noise explosion.
    
    Args:
        coefficients: Gabor coefficients
        transmission: Transmission map
        local_window: Window size for local statistics
    
    Returns:
        Adaptively corrected coefficients
    """
    logger.info("Applying adaptive correction based on local transmission")
    
    from scipy.ndimage import uniform_filter
    
    # Local mean of transmission
    t_local_mean = uniform_filter(transmission, size=local_window)
    
    # Adaptive epsilon: higher in foggy regions
    epsilon_adaptive = 0.1 + 0.3 * (1 - t_local_mean)
    
    # Safe transmission with adaptive floor
    t_safe = np.maximum(transmission, epsilon_adaptive)
    amplification = 1.0 / t_safe
    
    # Additional dampening in very foggy regions
    damping = np.clip(t_local_mean * 2, 0.3, 1.0)
    amplification *= damping
    
    corrected = []
    
    for coeff in coefficients:
        omega = coeff['omega']
        sigma = coeff['sigma']
        W = compute_weight_function(omega, sigma, alpha=2.0, beta=1.5)
        
        cx_corrected = coeff['cx'] * amplification * W
        cy_corrected = coeff['cy'] * amplification * W
        
        corrected.append({
            'cx': cx_corrected,
            'cy': cy_corrected,
            'theta': coeff['theta'],
            'sigma': sigma,
            'omega': omega,
            'kernel': coeff['kernel']
        })
    
    logger.info("Adaptive correction complete")
    return corrected
