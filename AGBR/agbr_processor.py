import numpy as np
import cv2
import logging
import time
from base.gradient_ops import compute_gradients, poisson_solve
from base.gabor_bank import create_gabor_bank, gabor_decompose, gabor_reconstruct

logger = logging.getLogger(__name__)

class AGBRProcessor:
    def __init__(
        self,
        patch_size=15,
        stride=4,
        omega=0.98,
        beta_high=0.7,
        beta_low=0.3,
        r_min=0.05,
        eps_thresh=1e-6,
        gabor_orientations=4,
        gabor_scales=[1.0, 2.0, 4.0, 6.0, 8.0],
        gabor_frequencies=[0.5, 1.0, 3.5, 5.0]
    ):
        self.k = patch_size
        self.stride = stride
        self.omega = omega
        self.beta_high = beta_high
        self.beta_low = beta_low
        self.r_min = r_min
        self.eps_thresh = eps_thresh
        
        self.gabor_orientations = gabor_orientations
        self.gabor_scales = gabor_scales
        self.gabor_frequencies = gabor_frequencies
        
        # Precompute Gabor bank
        self.gabor_bank = create_gabor_bank(
            orientations=gabor_orientations,
            scales=gabor_scales,
            frequencies=gabor_frequencies,
            ksize=patch_size * 2 + 1
        )
        logger.info(f"AGBRProcessor initialized: k={patch_size}, stride={stride}, omega={omega}")

    def process(self, image):
        """
        Full AGBR pipeline (memory optimized per-channel)
        """
        start_time = time.time()
        
        img_float = image.astype(np.float32) / 255.0
        h, w, c = img_float.shape
        restored_img = np.zeros_like(img_float)
        
        n_filters = len(self.gabor_bank)
        
        for ch in range(c):
            logger.info(f"Processing channel {ch+1}/{c}...")
            
            # Step 2: Gradients
            grad_x, grad_y = compute_gradients(img_float[:, :, ch])
            
            # Step 3: Gabor Decomposition
            logger.info(f"  Gabor decomposition...")
            ch_coeffs = gabor_decompose(grad_x, grad_y, self.gabor_bank)
            
            # Convert to numpy arrays for windowed processing
            cx_array = np.zeros((h, w, n_filters), dtype=np.float32)
            cy_array = np.zeros((h, w, n_filters), dtype=np.float32)
            for i in range(n_filters):
                cx_array[:, :, i] = ch_coeffs[i]['cx']
                cy_array[:, :, i] = ch_coeffs[i]['cy']
            
            # Step 4-6: Adaptive correction
            logger.info(f"  Adaptive correction (stride={self.stride})...")
            new_cx, new_cy = self._apply_adaptive_correction(cx_array, cy_array)
            
            # Free memory
            del cx_array
            del cy_array
            
            # Step 7: Reconstruction
            logger.info("  Reconstructing gradients...")
            final_coeffs = []
            for i in range(n_filters):
                d = ch_coeffs[i].copy()
                d['cx'] = new_cx[:, :, i]
                d['cy'] = new_cy[:, :, i]
                final_coeffs.append(d)
                
            rec_grad_x, rec_grad_y = gabor_reconstruct(final_coeffs)
            
            # Free memory
            del new_cx
            del new_cy
            del ch_coeffs
            
            # Step 8: Poisson Restore
            logger.info("  Solving Poisson equation...")
            restored_ch = poisson_solve(rec_grad_x, rec_grad_y)
            
            # Step 9: DC Component Correction
            mean_orig = np.mean(img_float[:, :, ch])
            mean_rec = np.mean(restored_ch)
            restored_ch += (mean_orig - mean_rec)
            
            restored_img[:, :, ch] = restored_ch
            
        # Step 10: Postprocessing
        restored_img = np.clip(restored_img, 0, 1)
        output = (restored_img * 255).astype(np.uint8)
        
        total_time = time.time() - start_time
        logger.info(f"Processing complete in {total_time:.2f} seconds")
        
        return output

    def _apply_adaptive_correction(self, cx, cy):
        """
        cx, cy shape: (H, W, NumFilters)
        """
        h, w, n_filters = cx.shape
        new_cx = cx.copy()
        new_cy = cy.copy()
        
        eps = 1e-8
        energy = cx**2 + cy**2
        
        for y in range(0, h - self.k + 1, self.stride):
            for x in range(0, w - self.k + 1, self.stride):
                patch_energy = energy[y:y+self.k, x:x+self.k] # (k, k, N)
                
                c_min = np.min(patch_energy)
                c_max = np.max(patch_energy)
                
                t_high = c_min + self.beta_high * (c_max - c_min)
                t_low = c_min + self.beta_low * (c_max - c_min)
                
                mask_high = patch_energy > t_high
                mask_low = patch_energy < t_low
                
                e_high = np.mean(patch_energy[mask_high]) if np.any(mask_high) else 0
                e_low = np.mean(patch_energy[mask_low]) if np.any(mask_low) else 0
                
                r = e_high / (e_high + e_low + eps)
                
                if r < self.r_min:
                    continue
                    
                range_y = slice(y, y + self.k)
                range_x = slice(x, x + self.k)
                
                pcx = new_cx[range_y, range_x, :]
                pcy = new_cy[range_y, range_x, :]
                
                mask_eps = patch_energy < self.eps_thresh
                pcx[mask_eps] = 0
                pcy[mask_eps] = 0
                
                pcx[mask_high] /= (r + eps)
                pcy[mask_high] /= (r + eps)
                pcx[mask_low] *= r
                pcy[mask_low] *= r
                
                new_cx[range_y, range_x, :] = pcx
                new_cy[range_y, range_x, :] = pcy
                    
        return new_cx, new_cy
