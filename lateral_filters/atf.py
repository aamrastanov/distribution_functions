import numpy as np
import cv2
from abf import AdaptiveBilateralFilter

class AdaptiveTrilateralFilter(AdaptiveBilateralFilter):
    def __init__(self, sigma_s=2.0, window_size=5):
        super().__init__(sigma_s, window_size)

    def calculate_road(self, image):
        """
        Calculates ROAD (Rank Order Absolute Difference) statistic.
        """
        print("Calculating ROAD statistics...")
        # ROAD usually uses 3x3 window, separate from filter window size
        road_window = 3 
        pad = road_window // 2
        
        if len(image.shape) == 3:
             # ROAD on grayscale usually, or max of channels. 
             # Let's do grayscale for detection map.
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        gray_float = gray.astype(np.float32)
        padded = cv2.copyMakeBorder(gray_float, pad, pad, pad, pad, cv2.BORDER_REFLECT)
        h, w = gray.shape
        
        road_map = np.zeros_like(gray, dtype=np.float32)
        
        for i in range(h):
            for j in range(w):
                patch = padded[i:i+road_window, j:j+road_window]
                center_val = padded[i+pad, j+pad]
                
                # Absolute differences
                diffs = np.abs(patch - center_val)
                # Exclude center (0)
                diffs_flat = np.sort(diffs.flatten())[1:] 
                
                # Sum 4 smallest values (m=4 typical for 3x3)
                road_val = np.sum(diffs_flat[:4])
                road_map[i, j] = road_val
                
        return road_map

    def apply(self, image):
        if self.sigma_r_lut is None:
            self.estimate_noise_profile(image)
        
        road_map = self.calculate_road(image)
        
        # Estimate sigma_I (mean of ROAD)
        sigma_i = np.mean(road_map)
        # Check for user-defined or adaptive threshold. Paper says mean is good approximation.
        sigma_j = sigma_i # Paper mentions sigma_j controls shape of J
        
        print(f"Applying Adaptive Trilateral Filter (Sigma_I={sigma_i:.2f})...")
        
        result = np.zeros_like(image, dtype=np.float32)
        pad = self.window_size // 2
        
        # Consistent padding
        if len(image.shape) == 3:
            padded_img = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT)
        else:
            padded_img = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT)

        padded_road = cv2.copyMakeBorder(road_map, pad, pad, pad, pad, cv2.BORDER_REFLECT)

        # Spatial weights Pre-calc
        x, y = np.meshgrid(np.arange(-pad, pad+1), np.arange(-pad, pad+1))
        spatial_weights = np.exp(-(x**2 + y**2) / (2 * self.sigma_s**2))

        h, w = image.shape[:2]

        for i in range(h):
            if i % 50 == 0: print(f"Row {i}/{h}")
            for j in range(w):
                cp_y, cp_x = i + pad, j + pad
                
                # Windows
                window_img = padded_img[i:i+self.window_size, j:j+self.window_size]
                window_road = padded_road[i:i+self.window_size, j:j+self.window_size]
                
                # Current Pixel Props
                if len(image.shape) == 3:
                     center_intensity = int(np.mean(image[i, j]))
                else:
                     center_intensity = int(image[i, j])
                     
                sigma_r = self.sigma_r_lut[center_intensity]
                if sigma_r < 1e-5: sigma_r = 1.0
                
                center_road = road_map[i, j]

                # Radiometric Weight (Standard Bilateral Component)
                diff = window_img.astype(np.float32) - padded_img[cp_y, cp_x].astype(np.float32)
                if len(image.shape) == 3:
                     dist_sq = np.sum(diff**2, axis=2)
                else:
                     dist_sq = diff**2
                w_r = np.exp(-dist_sq / (2 * sigma_r**2))
                
                # Impulsive Weight
                w_i = np.exp(-(window_road**2) / (2 * sigma_i**2))
                
                # Joint Impulsivity J(p, q)
                # J depends on (ROAD(p) + ROAD(q))/2
                avg_road = (center_road + window_road) / 2.0
                j_vals = 1.0 - np.exp(-(avg_road**2) / (2 * sigma_j**2))
                
                # Combined Weight: w = w_S * w_R^(1-J) * w_I^J
                # Using power here per element
                # Avoid potential domain errors if w_R is 0
                w_r_safe = np.maximum(w_r, 1e-10)
                w_i_safe = np.maximum(w_i, 1e-10)
                
                combined_weights = spatial_weights * (w_r_safe ** (1.0 - j_vals)) * (w_i_safe ** j_vals)
                
                norm = np.sum(combined_weights)
                
                if len(image.shape) == 3:
                     weights_expanded = combined_weights[:, :, np.newaxis]
                     result[i, j] = np.sum(window_img * weights_expanded, axis=(0,1)) / norm
                else:
                     result[i, j] = np.sum(window_img * combined_weights) / norm

        return np.clip(result, 0, 255).astype(np.uint8)
