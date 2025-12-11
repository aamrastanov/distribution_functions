import numpy as np
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def calculate_smoothness(matrix, freq_limit):
    """
    Calculates the smoothness of a 2D matrix based on the ratio of low-frequency energy.
    
    Args:
        matrix (np.ndarray): Input 2D matrix.
        freq_limit (float): Frequency radius limit (normalized 0.0 to 0.5).
                           Frequencies below this limit are considered 'low'.
                           
    Returns:
        float: Smoothness score (0.0 to 1.0).
    """
    if matrix.ndim != 2:
        raise ValueError("Input matrix must be 2D")
        
    rows, cols = matrix.shape
    
    # 1. Perform 2D FFT
    f_transform = np.fft.fft2(matrix)
    
    # 2. Shift zero frequency to center
    f_shift = np.fft.fftshift(f_transform)
    
    # 3. Calculate Power Spectrum
    power_spectrum = np.abs(f_shift)**2
    total_power = np.sum(power_spectrum)
    
    if total_power == 0:
        return 0.0
        
    # 4. Generate frequency coordinates (centered)
    # fftfreq returns [0, 1/n, ..., -1/n]. Shifted it becomes [-0.5, ..., 0, ..., 0.5]
    freq_y = np.fft.fftshift(np.fft.fftfreq(rows))
    freq_x = np.fft.fftshift(np.fft.fftfreq(cols))
    
    # Create grid of frequencies
    # Note: meshgrid with 'ij' indexing to match matrix indexing
    fy, fx = np.meshgrid(freq_y, freq_x, indexing='ij')
    
    # 5. Calculate radius for each point
    freq_radius = np.sqrt(fy**2 + fx**2)
    
    # 6. Select low frequencies
    mask = freq_radius < freq_limit
    
    low_freq_power = np.sum(power_spectrum[mask])
    
    # 7. Calculate ratio
    smoothness = low_freq_power / total_power
    
    return smoothness

def main():
    logger.info("Testing calculate_smoothness function...")
    
    # Test 1: Random Noise (High frequency content)
    # Shape 100x100
    noise = np.random.rand(100, 100)
    limit = 0.2
    score_noise = calculate_smoothness(noise, freq_limit=limit)
    logger.info(f"Random Noise Smoothness (limit={limit}): {score_noise:.4f}")
    
    # Test 2: Smooth Gaussian
    y, x = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100), indexing='ij')
    gaussian = np.exp(-(x**2 + y**2) / 2.0)
    score_gaussian = calculate_smoothness(gaussian, freq_limit=limit)
    logger.info(f"Gaussian Smoothness (limit={limit}): {score_gaussian:.4f}")
    
    if score_gaussian > score_noise:
        logger.info("SUCCESS: Gaussian is smoother than noise.")
    else:
        logger.warning("FAILURE: Gaussian score is not higher than noise!")

if __name__ == "__main__":
    main()
