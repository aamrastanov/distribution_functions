"""
GBR Processor - Main processing class for Garbon Based Restore algorithm.

Combines all 6 steps into a unified pipeline.
"""

import numpy as np
import cv2
import logging
from datetime import datetime

from base.dark_channel import (
    compute_dark_channel,
    estimate_atmospheric_light,
    estimate_transmission,
    refine_transmission
)
from base.gradient_ops import (
    compute_gradients_color,
    poisson_solve
)
from base.gabor_bank import (
    create_gabor_bank,
    gabor_decompose_color,
    gabor_reconstruct_color
)
from correction import (
    correct_coefficients_color
)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class GBRProcessor:
    """
    Garbon Based Restore (GBR) image restoration processor.
    
    Combines Dark Channel Prior with Gabor decomposition for
    adaptive contrast restoration in degraded images.
    """
    
    def __init__(self, 
                 patch_size: int = 15,
                 omega: float = 0.95,
                 guided_radius: int = 60,
                 guided_eps: float = 1e-3,
                 gabor_orientations: int = 8,
                 gabor_scales: list = None,
                 gabor_frequencies: list = None,
                 gabor_ksize: int = 31,
                 correction_epsilon: float = 0.1,
                 correction_alpha: float = 2.0,
                 correction_beta: float = 1.5,
                 noise_threshold: float = 0.01,
                 transmission_min: float = 0.1,
                 top_percent: float = 0.001,
                 scale_limit: float = 2.0):
        """
        Initialize GBR processor with parameters.
        
        Args:
            patch_size: Dark channel patch size
            omega: Haze preservation coefficient
            guided_radius: Guided filter radius
            guided_eps: Guided filter regularization
            gabor_orientations: Number of Gabor orientations
            gabor_scales: Gabor sigma values
            gabor_frequencies: Gabor frequencies
            gabor_ksize: Kernel size for Gabor filters
            correction_epsilon: Min transmission for correction
            correction_alpha: Weight function alpha
            correction_beta: Weight function beta
            noise_threshold: Noise suppression threshold
            transmission_min: Minimum transmission floor
            top_percent: Top percentage of pixels for atmospheric light estimation
            scale_limit: Maximum scaling factor for contrast in Poisson integration
        """
        self.patch_size = patch_size
        self.omega = omega
        self.guided_radius = guided_radius
        self.guided_eps = guided_eps
        self.gabor_orientations = gabor_orientations
        self.gabor_scales = gabor_scales or [2.0, 4.0, 8.0]
        self.gabor_frequencies = gabor_frequencies or [0.1, 0.2, 0.3]
        self.gabor_ksize = gabor_ksize
        self.correction_epsilon = correction_epsilon
        self.correction_alpha = correction_alpha
        self.correction_beta = correction_beta
        self.noise_threshold = noise_threshold
        self.transmission_min = transmission_min
        self.top_percent = top_percent
        self.scale_limit = scale_limit
        
        # Will be set during processing
        self.dark_channel = None
        self.atmospheric_light = None
        self.transmission = None
        self.transmission_refined = None
        self.gabor_bank = None
        
        logger.info("GBR Processor initialized")
        logger.info(f"  Patch size: {patch_size}")
        logger.info(f"  Gabor orientations: {gabor_orientations}")
        logger.info(f"  Gabor scales: {self.gabor_scales}")
        logger.info(f"  Gabor frequencies: {self.gabor_frequencies}")
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Run full GBR pipeline on input image.
        
        Args:
            image: Input BGR image (H, W, 3)
        
        Returns:
            Restored image (H, W, 3)
        """
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("Starting GBR Processing Pipeline")
        logger.info(f"Input image shape: {image.shape}")
        logger.info("=" * 60)
        
        # ========================================
        # STEP 1: Dark Channel Parameter Estimation
        # ========================================
        logger.info("\n[STEP 1/6] Estimating atmospheric parameters (Dark Channel)")
        step_start = datetime.now()
        
        self.dark_channel = compute_dark_channel(image, self.patch_size)
        self.atmospheric_light = estimate_atmospheric_light(image, self.dark_channel, self.top_percent)
        self.transmission = estimate_transmission(
            image, self.atmospheric_light, self.omega, self.patch_size, self.transmission_min
        )
        self.transmission_refined = refine_transmission(
            image, self.transmission, self.guided_radius, self.guided_eps, self.transmission_min
        )
        
        logger.info(f"Step 1 completed in {(datetime.now() - step_start).total_seconds():.2f}s")
        
        # ========================================
        # STEP 2: Gradient Computation
        # ========================================
        logger.info("\n[STEP 2/6] Computing image gradients")
        step_start = datetime.now()
        
        gx, gy = compute_gradients_color(image)
        
        logger.info(f"Step 2 completed in {(datetime.now() - step_start).total_seconds():.2f}s")
        
        # ========================================
        # STEP 3: Gabor Decomposition
        # ========================================
        logger.info("\n[STEP 3/6] Gabor decomposition")
        step_start = datetime.now()
        
        self.gabor_bank = create_gabor_bank(
            self.gabor_orientations,
            self.gabor_scales,
            self.gabor_frequencies,
            ksize=self.gabor_ksize
        )
        
        coefficients = gabor_decompose_color(gx, gy, self.gabor_bank)
        
        logger.info(f"Step 3 completed in {(datetime.now() - step_start).total_seconds():.2f}s")
        
        # ========================================
        # STEP 4: Physical Correction
        # ========================================
        logger.info("\n[STEP 4/6] Physical correction of coefficients")
        step_start = datetime.now()
        
        corrected_coeffs = correct_coefficients_color(
            coefficients, 
            self.transmission_refined,
            epsilon=self.correction_epsilon,
            alpha=self.correction_alpha,
            beta=self.correction_beta,
            noise_threshold=self.noise_threshold
        )
        
        logger.info(f"Step 4 completed in {(datetime.now() - step_start).total_seconds():.2f}s")
        
        # ========================================
        # STEP 5: Gradient Reconstruction
        # ========================================
        logger.info("\n[STEP 5/6] Reconstructing gradient field")
        step_start = datetime.now()
        
        gx_hat, gy_hat = gabor_reconstruct_color(corrected_coeffs)
        
        logger.info(f"Step 5 completed in {(datetime.now() - step_start).total_seconds():.2f}s")
        
        # ========================================
        # STEP 6: Poisson Integration
        # ========================================
        logger.info("\n[STEP 6/6] Solving Poisson equation")
        step_start = datetime.now()
        
        restored = poisson_solve(gx_hat, gy_hat)
        
        # ========================================
        # STEP 9: DC Component Correction & Contrast Scaling
        # ========================================
        logger.info("\n[STEP 9/6] DC Component Correction & Contrast Scaling")
        step_start = datetime.now()
        
        # Formula: I_final = (I_rec - I_rec_mean) * scale_limit + I_orig_mean
        result = np.zeros_like(image, dtype=np.float64)
        
        for c in range(3):
            orig_channel = image[:, :, c].astype(np.float64)
            rec_channel = restored[:, :, c]
            
            orig_mean = np.mean(orig_channel)
            rec_mean = np.mean(rec_channel)
            
            # Center the reconstructed channel (remove its DC component)
            channel_centered = rec_channel - rec_mean
            
            # Use scale_limit as a direct multiplier relative to original contrast.
            # If scale_limit=1.0, the output contrast matches the original image.
            # If scale_limit > 1.0, the output contrast will be higher than the original.
            orig_std = np.std(orig_channel)
            rec_std = np.std(channel_centered)
            
            if rec_std > 1e-6:
                # Direct linear mapping: matches original when scale_limit=1
                scale = self.scale_limit * (orig_std / rec_std)
            else:
                scale = self.scale_limit
                
            channel_scaled = channel_centered * scale
            
            # Add back original mean (DC anchoring)
            channel_final = channel_scaled + orig_mean
            
            result[:, :, c] = channel_final
            
        # Clip to valid range
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        logger.info(f"Step 9 completed in {(datetime.now() - step_start).total_seconds():.2f}s")
        
        # ========================================
        # COMPLETE
        # ========================================
        total_time = (datetime.now() - start_time).total_seconds()
        logger.info("\n" + "=" * 60)
        logger.info(f"GBR Processing Complete!")
        logger.info(f"Total processing time: {total_time:.2f} seconds")
        logger.info(f"Output image shape: {result.shape}")
        logger.info("=" * 60)
        
        return result
    
    def process_dehaze(self, image: np.ndarray, detail_boost: float = 0.3) -> np.ndarray:
        """
        Alternative processing: Direct dehazing with optional detail enhancement.
        
        This approach:
        1. Uses Dark Channel Prior to estimate transmission
        2. Applies standard dehazing formula: J = (I - A) / t + A
        3. Optionally enhances details using Gabor-based gradient boosting
        
        Args:
            image: Input BGR image
            detail_boost: Amount of Gabor-based detail enhancement (0-1)
        
        Returns:
            Dehazed and enhanced image
        """
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("Starting GBR Processing (Dehaze Mode)")
        logger.info(f"Input image shape: {image.shape}")
        logger.info(f"Detail boost: {detail_boost}")
        logger.info("=" * 60)
        
        # ========================================
        # STEP 1: Dark Channel Parameter Estimation
        # ========================================
        logger.info("\n[STEP 1/4] Estimating atmospheric parameters")
        step_start = datetime.now()
        
        self.dark_channel = compute_dark_channel(image, self.patch_size)
        self.atmospheric_light = estimate_atmospheric_light(image, self.dark_channel)
        self.transmission = estimate_transmission(
            image, self.atmospheric_light, self.omega, self.patch_size
        )
        self.transmission_refined = refine_transmission(
            image, self.transmission, self.guided_radius, self.guided_eps
        )
        
        logger.info(f"Step 1 completed in {(datetime.now() - step_start).total_seconds():.2f}s")
        
        # ========================================
        # STEP 2: Apply Dehazing Formula
        # ========================================
        logger.info("\n[STEP 2/4] Applying dehazing formula")
        step_start = datetime.now()
        
        # J = (I - A) / max(t, t0) + A
        t0 = 0.1  # Minimum transmission to avoid division issues
        
        A = self.atmospheric_light.reshape(1, 1, 3)
        t = np.maximum(self.transmission_refined, t0)[:, :, np.newaxis]
        
        # Dehazing
        dehazed = (image.astype(np.float64) - A) / t + A
        dehazed = np.clip(dehazed, 0, 255)
        
        logger.info(f"Dehazed image range: [{dehazed.min():.1f}, {dehazed.max():.1f}]")
        logger.info(f"Step 2 completed in {(datetime.now() - step_start).total_seconds():.2f}s")
        
        # ========================================
        # STEP 3: Detail Enhancement (Optional)
        # ========================================
        if detail_boost > 0:
            logger.info(f"\n[STEP 3/4] Enhancing details with Gabor (boost={detail_boost})")
            step_start = datetime.now()
            
            # Compute gradients of dehazed image
            gx, gy = compute_gradients_color(dehazed.astype(np.uint8))
            
            # Create smaller Gabor bank for detail enhancement
            detail_bank = create_gabor_bank(
                orientations=4,  # Fewer orientations for speed
                scales=[2.0, 4.0],
                frequencies=[0.15, 0.25]
            )
            
            # Decompose
            coeffs = gabor_decompose_color(gx, gy, detail_bank)
            
            # Apply mild correction based on transmission
            corrected = correct_coefficients_color(
                coeffs, 
                self.transmission_refined,
                epsilon=0.2,  # Higher epsilon = less aggressive
                alpha=1.5,    # Lower alpha = less frequency-dependent damping
                beta=1.0,
                noise_threshold=0.02
            )
            
            # Reconstruct detail layer
            gx_hat, gy_hat = gabor_reconstruct_color(corrected)
            
            # Compute detail difference
            detail_x = gx_hat - gx
            detail_y = gy_hat - gy
            
            # Add enhanced details back to dehazed image
            for c in range(3):
                # Combine gradient differences as detail
                detail_combined = detail_x[:, :, c] + detail_y[:, :, c]
                # Apply as direct detail enhancement (no Laplacian needed)
                dehazed[:, :, c] += detail_boost * detail_combined * 0.1
            
            dehazed = np.clip(dehazed, 0, 255)
            logger.info(f"Step 3 completed in {(datetime.now() - step_start).total_seconds():.2f}s")
        else:
            logger.info("\n[STEP 3/4] Skipping detail enhancement (boost=0)")
        
        # ========================================
        # STEP 4: Post-processing
        # ========================================
        logger.info("\n[STEP 4/4] Post-processing")
        step_start = datetime.now()
        
        result = dehazed.astype(np.uint8)
        
        # Optional: Apply mild CLAHE for local contrast
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        logger.info(f"Step 4 completed in {(datetime.now() - step_start).total_seconds():.2f}s")
        
        # ========================================
        # COMPLETE
        # ========================================
        total_time = (datetime.now() - start_time).total_seconds()
        logger.info("\n" + "=" * 60)
        logger.info(f"GBR Dehaze Processing Complete!")
        logger.info(f"Total processing time: {total_time:.2f} seconds")
        logger.info("=" * 60)
        
        return result
    
    def get_intermediate_results(self) -> dict:
        """
        Get intermediate processing results for visualization/debugging.
        
        Returns:
            Dict with dark_channel, transmission, etc.
        """
        return {
            'dark_channel': self.dark_channel,
            'atmospheric_light': self.atmospheric_light,
            'transmission_raw': self.transmission,
            'transmission_refined': self.transmission_refined
        }


def enhance_with_original(restored: np.ndarray, original: np.ndarray, 
                          alpha: float = 0.3) -> np.ndarray:
    """
    Blend restored image with original for natural look.
    
    Args:
        restored: GBR restored image
        original: Original input image
        alpha: Blend factor (0 = all restored, 1 = all original)
    
    Returns:
        Blended image
    """
    logger.info(f"Blending restored with original (alpha={alpha})")
    
    blended = (1 - alpha) * restored.astype(np.float64) + alpha * original.astype(np.float64)
    return np.clip(blended, 0, 255).astype(np.uint8)
