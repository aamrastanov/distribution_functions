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
        patch_size=51,
        stride=5,
        block_size=5,
        omega=0.98,
        beta_high=0.7,
        beta_low=0.3,
        r_min=0.05,
        eps_thresh=1e-6,
        gabor_scales=[1.0, 2.0, 4.0, 6.0, 7.0],
        gabor_frequencies=[0.5, 1.0, 3.5, 5.0]
    ):
        self.patch_size = patch_size # window size 51
        self.stride = stride # step 5
        self.block_size = block_size # middle block 5x5
        self.omega = omega
        self.beta_high = beta_high
        self.beta_low = beta_low
        self.r_min = r_min
        self.eps_thresh = eps_thresh
        
        self.gabor_scales = gabor_scales
        self.gabor_frequencies = gabor_frequencies
        
        # Precompute 1D Gabor basis matrix
        self.gabor_basis = self._precompute_1d_gabor_basis()
        
        # Precompute basis for central block reconstruction
        offset = self.patch_size // 2 - self.block_size // 2
        self.b_range = slice(offset, offset + self.block_size)
        self.basis_b = self.gabor_basis[:, self.b_range]
        
        # Calibrate reconstruction scale
        test_impulse = np.zeros(self.patch_size, dtype=np.float32)
        test_impulse[self.patch_size // 2] = 1.0
        test_coeffs = test_impulse @ self.gabor_basis.T
        test_rec = test_coeffs @ self.gabor_basis
        self.reconstruction_scale = test_rec[self.patch_size // 2]
        
        logger.info(f"AGBRProcessor initialized: window={patch_size}, S={stride}, scale={self.reconstruction_scale:.2f}")

    def _precompute_1d_gabor_basis(self):
        """
        Compute 1D Gabor basis functions for a window.
        Returns a matrix of size (N_filters, window_size).
        """
        half_w = self.patch_size // 2
        t = np.linspace(-half_w, half_w, self.patch_size)
        basis = []
        
        for sigma in self.gabor_scales:
            for freq in self.gabor_frequencies:
                # Gaussian envelope
                envelope = np.exp(-t**2 / (2 * sigma**2))
                g_cos = envelope * np.cos(2 * np.pi * freq * t / self.patch_size)
                g_sin = envelope * np.sin(2 * np.pi * freq * t / self.patch_size)
                
                norm_cos = np.linalg.norm(g_cos)
                norm_sin = np.linalg.norm(g_sin)
                
                if norm_cos > 1e-10: basis.append(g_cos / norm_cos)
                if norm_sin > 1e-10: basis.append(g_sin / norm_sin)
                
        return np.array(basis, dtype=np.float32)

    def process(self, image):
        """
        Full AGBR pipeline with windowed processing (optimized).
        """
        start_time = time.time()
        
        img_float = image.astype(np.float32) / 255.0
        h, w, c = img_float.shape
        restored_img = np.zeros_like(img_float)
        
        # Center block offsets (for 51x51 window and 5x5 block)
        offset = self.patch_size // 2 - self.block_size // 2
        
        for ch in range(c):
            logger.info(f"Processing channel {ch+1}/{c}...")
            
            # Step 2: Gradients
            gx, gy = compute_gradients(img_float[:, :, ch])
            
            # Prepare new gradient matrices (copy original for boundaries)
            new_gx = gx.copy()
            new_gy = gy.copy()
            
            # Step 3-7: Windowed Gabor processing
            logger.info(f"  Windowed Gabor processing (stride={self.stride})...")
            
            count = 0
            for y in range(0, h - self.patch_size + 1, self.stride):
                for x in range(0, w - self.patch_size + 1, self.stride):
                    # Extract 51x51 window
                    px = gx[y : y + self.patch_size, x : x + self.patch_size]
                    py = gy[y : y + self.patch_size, x : x + self.patch_size]
                    
                    # Process window (Optimized: returns only 5x5 blocks)
                    block_x, rx = self._process_patch(px)
                    block_y_t, ry = self._process_patch(py.T)
                    block_y = block_y_t.T
                    
                    if count % 5000 == 0:
                        logger.info(f"    Window {count:5d}: rx={rx:.3f}, ry={ry:.3f}")
                    
                    # Replace 5x5 middle values
                    gy_start = y + offset
                    gx_start = x + offset
                    new_gx[gy_start : gy_start + self.block_size, 
                           gx_start : gx_start + self.block_size] = block_x
                    new_gy[gy_start : gy_start + self.block_size, 
                           gx_start : gx_start + self.block_size] = block_y
                    count += 1
            
            logger.info(f"  Processed {count} windows.")
            
            # Step 8: Poisson Restore
            logger.info("  Solving Poisson equation...")
            restored_ch = poisson_solve(new_gx, new_gy)
            
            # Step 9: Brightness matching
            mean_orig = np.mean(img_float[:, :, ch])
            mean_rec = np.mean(restored_ch)
            restored_ch += (mean_orig - mean_rec)
            
            restored_img[:, :, ch] = restored_ch
            
        restored_img = np.clip(restored_img, 0, 1)
        output = (restored_img * 255).astype(np.uint8)
        
        total_time = time.time() - start_time
        logger.info(f"Processing complete in {total_time:.2f} seconds")
        
        return output

    def _correct_coefficients(self, coeffs):
        """
        Apply non-linear correction to Gabor coefficients for detail boost and noise suppression.
        """
        energies = coeffs**2
        c_min, c_max = np.min(energies), np.max(energies)
        t_high = c_min + self.beta_high * (c_max - c_min)
        t_low = c_min + self.beta_low * (c_max - c_min)
        
        # mask_high = energies > t_high
        mask_low = energies < t_low
        
        # Use all available coefficients for statistics
        sum_all = np.sum(energies) + 1e-12
        sum_low = np.sum(energies[mask_low])
        r = 1.0 - self.omega * (sum_low / sum_all)
        
        # Copy for modification
        c_b = coeffs.copy()
        
        if r >= self.r_min:
            m_high = energies > t_high
            m_low = energies < t_low
            
            c_b[m_high] /= (r + 1e-10) # Detail boost
            c_b[m_low] *= r           # Noise suppression
        
        # Universal noise floor cutoff
        c_b[c_b**2 < self.eps_thresh] = 0
            
        return c_b, r

    def _process_patch(self, p):
        """
        Process a single 1D-oriented patch (horizontal signal).
        Optimized: takes only the rows needed for central block.
        """
        # Take only central rows (e.g., 5 rows instead of 51)
        # Resulting shape: (block_size, patch_size)
        p_slice = p[self.b_range, :]
        
        # 1D Decomposition (block_size x N_filters)
        coeffs = p_slice @ self.gabor_basis.T
        
        # Apply correction (stats are now calculated only on these central rows)
        c_b, r = self._correct_coefficients(coeffs)
        
        # Reconstruction (block_size x N_filters * N_filters x block_size -> block_size x block_size)
        block = (c_b @ self.basis_b) / self.reconstruction_scale
        
        return block, r
