import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops

class AdaptiveBilateralFilter:
    """
    Adaptive Bilateral Filter (ABF) as described in JoG_Oct_2021_denoising.pdf.
    Uses adaptive sigma_r estimation based on GLCM homogeneity.
    """
    def __init__(self, sigma_s=2.0, window_size=5):
        self.sigma_s = sigma_s
        self.window_size = window_size
        self.sigma_r_lut = None

    def estimate_noise_profile(self, image):
        """
        Estimates the noise profile (sigma_r) for different intensity levels
        using GLCM homogeneity as described in the paper.
        """
        print("Estimating noise profile using GLCM...")
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        h, w = gray.shape
        
        # Lists to store means and std_devs of homogeneous patches
        homogeneous_means = []
        homogeneous_stds = []

        # Sliding window with stride for speed
        stride = 5 
        
        for y in range(0, h - self.window_size + 1, stride):
            for x in range(0, w - self.window_size + 1, stride):
                patch = gray[y:y+self.window_size, x:x+self.window_size]
                
                # Calculate GLCM Homogeneity
                # Paper says displacement vector d=(1,1), which is angle 45 degrees
                glcm = graycomatrix(patch, distances=[1], angles=[np.pi/4], 
                                   levels=256, symmetric=True, normed=True)
                homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
                
                # Threshold from paper: 0.99
                if homogeneity > 0.99:
                    mean_val = np.mean(patch)
                    std_val = np.std(patch)
                    homogeneous_means.append(mean_val)
                    homogeneous_stds.append(std_val)

        if not homogeneous_means:
            print("Warning: No homogeneous regions found. Defaulting to fixed sigma_r.")
            self.sigma_r_lut = np.full(256, 15.0)  # Fallback
            return

        print(f"Found {len(homogeneous_means)} homogeneous patches")
        
        # Create LUT - bin means into groups of 5 intensity levels
        bins = np.arange(0, 260, 5)  # 0, 5, ..., 255
        digitized = np.digitize(homogeneous_means, bins)
        
        std_sums = np.zeros(len(bins))
        std_counts = np.zeros(len(bins))
        
        for i, bin_idx in enumerate(digitized):
            if bin_idx < len(bins):
                std_sums[bin_idx] += homogeneous_stds[i]
                std_counts[bin_idx] += 1
                
        # Calculate average std for each bin
        bin_stds = np.zeros(len(bins))
        for i in range(len(bins)):
            if std_counts[i] > 0:
                bin_stds[i] = std_sums[i] / std_counts[i]
            else:
                bin_stds[i] = np.nan  # Mark for interpolation

        # Interpolate missing values
        valid_indices = np.where(~np.isnan(bin_stds))[0]
        if len(valid_indices) == 0:
            self.sigma_r_lut = np.full(256, 15.0)
            return

        interp_stds = np.interp(np.arange(len(bins)), valid_indices, bin_stds[valid_indices])
        
        # Expand bins to full LUT (256 entries)
        self.sigma_r_lut = np.zeros(256)
        for i in range(256):
            bin_idx = i // 5
            if bin_idx < len(interp_stds):
                self.sigma_r_lut[i] = max(interp_stds[bin_idx], 1.0)  # Minimum sigma_r = 1
            else:
                self.sigma_r_lut[i] = max(interp_stds[-1], 1.0)
        
        print(f"Sigma_R range: [{self.sigma_r_lut.min():.2f}, {self.sigma_r_lut.max():.2f}]")

    def apply(self, image):
        """Apply the Adaptive Bilateral Filter to the image."""
        if self.sigma_r_lut is None:
            self.estimate_noise_profile(image)

        print("Applying Adaptive Bilateral Filter...")
        result = np.zeros_like(image, dtype=np.float32)
        pad = self.window_size // 2
        
        # Pad image for boundary handling
        padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT)

        # Pre-compute spatial weights
        x, y = np.meshgrid(np.arange(-pad, pad+1), np.arange(-pad, pad+1))
        spatial_weights = np.exp(-(x**2 + y**2) / (2 * self.sigma_s**2))

        h, w = image.shape[:2]
        
        # Apply filter (slow Python loop - could be optimized with numba)
        for i in range(h):
            if i % 100 == 0:
                print(f"Progress: {i}/{h} rows ({100*i//h}%)")
            for j in range(w):
                # Center pixel (in padded coords)
                cp_y, cp_x = i + pad, j + pad
                
                # Neighborhood (window)
                window = padded[i:i+self.window_size, j:j+self.window_size]
                
                # Get sigma_r for center pixel
                if len(image.shape) == 3:
                    center_intensity = int(np.mean(image[i, j]))
                else:
                    center_intensity = int(image[i, j])
                
                center_intensity = np.clip(center_intensity, 0, 255)
                sigma_r = self.sigma_r_lut[center_intensity]
                
                # Radiometric weights: Exp(-(Ip - Iq)^2 / (2 * sigma_r^2))
                diff = window.astype(np.float32) - padded[cp_y, cp_x].astype(np.float32)
                
                # For RGB, use Euclidean distance in color space
                if len(image.shape) == 3:
                    dist_sq = np.sum(diff**2, axis=2)
                else:
                    dist_sq = diff**2
                     
                radiometric_weights = np.exp(-dist_sq / (2 * sigma_r**2))
                
                weights = spatial_weights * radiometric_weights
                norm = np.sum(weights)
                
                if norm < 1e-10:
                    result[i, j] = image[i, j]
                    continue
                
                if len(image.shape) == 3:
                    # Broadcast weights for RGB
                    weights_expanded = weights[:, :, np.newaxis]
                    result[i, j] = np.sum(window * weights_expanded, axis=(0, 1)) / norm
                else:
                    result[i, j] = np.sum(window * weights) / norm

        return np.clip(result, 0, 255).astype(np.uint8)
