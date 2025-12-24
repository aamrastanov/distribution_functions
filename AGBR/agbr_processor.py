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
        eps_thresh=1e-6,
        gabor_scales=[1.0, 2.0, 4.0, 6.0, 7.0],
        gabor_frequencies=[0.5, 1.0, 3.5, 5.0],
        scattered_signal_suppress_coef=0.1,
        main_signal_ampligy_coef=0.5
    ):
        self.patch_size = patch_size # window size 51
        self.stride = stride # step 5
        self.block_size = block_size # middle block 5x5
        self.eps_thresh = eps_thresh
        self.scattered_signal_suppress_coef = scattered_signal_suppress_coef
        self.main_signal_ampligy_coef = main_signal_ampligy_coef
        
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
        Strict order: Scale -> Frequency -> Phase (Cos, Sin)
        """
        half_w = self.patch_size // 2
        t = np.linspace(-half_w, half_w, self.patch_size)
        basis = []
        
        for sigma in self.gabor_scales:
            for freq in self.gabor_frequencies:
                # Gaussian envelope
                envelope = np.exp(-t**2 / (2 * sigma**2))
                # Wavelength based on cycles per patch
                lambd = self.patch_size / freq
                
                g_cos = envelope * np.cos(2 * np.pi * t / lambd)
                g_sin = envelope * np.sin(2 * np.pi * t / lambd)
                
                norm_cos = np.linalg.norm(g_cos)
                norm_sin = np.linalg.norm(g_sin)
                
                # We append even if norm is low to keep the index mapping consistent
                basis.append(g_cos / (norm_cos + 1e-12))
                basis.append(g_sin / (norm_sin + 1e-12))
                
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
                    block_x, ex, stats_x = self._process_patch(px)
                    block_y_t, ey, stats_y = self._process_patch(py.T)
                    block_y = block_y_t.T
                    
                    if count % 5000 == 0:
                        logger.info(f"    Window {count:5d}: E_trans_x={ex:.4f}, E_trans_y={ey:.4f}")
                        # Log top contributors to energy transfer
                        top_x = sorted(stats_x.items(), key=lambda x: x[1], reverse=True)[:3]
                        msg_x = ", ".join([f"S{s}/F{f}: {e:.4f}" for (s, f), e in top_x])
                        logger.info(f"      Top X source: {msg_x}")
                    
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
        Redistribute energy from large-scale (scattering) to small-scale (main) components.
        coeffs: (block_size, N_filters)
        """
        n_scales = len(self.gabor_scales)
        n_freqs = len(self.gabor_frequencies)
        n_per_scale = n_freqs * 2 # 2 phases per freq
        
        c_new = coeffs.copy()
        
        # Scaling factors
        s_suppress = self.scattered_signal_suppress_coef
        m_amplify = self.main_signal_ampligy_coef
        
        # Helper to get index
        def get_idx(s_idx, f_idx, p_idx):
            return s_idx * n_per_scale + f_idx * 2 + p_idx

        # Large-scale "scattering" components to be suppressed: Last two scales
        scattering_indices = [n_scales - 2, n_scales - 1]
        # Small-scale "main" components to be boosted: First two scales
        main_indices = [0, 1]
        
        total_energy_transferred = 0
        energy_breakdown = {} # (sigma, freq) -> total energy taken
        
        for f_idx in range(n_freqs):
            for p_idx in [0, 1]:
                # Collect energy to be redistributed from large scales
                energy_to_transfer = 0
                for s_scat in scattering_indices:
                    idx_scat = get_idx(s_scat, f_idx, p_idx)
                    val_scat = c_new[:, idx_scat]
                    
                    # 1. Suppress scattering signal
                    # c_scat_new = sqrt(1 - suppress) * c_scat
                    c_new[:, idx_scat] *= np.sqrt(1.0 - s_suppress)
                    
                    # Energy taken: suppress * val^2
                    energy_taken = s_suppress * (val_scat**2)
                    energy_to_transfer += energy_taken
                    
                    # Track breakdown
                    sigma = self.gabor_scales[s_scat]
                    freq = self.gabor_frequencies[f_idx]
                    key = (sigma, freq)
                    energy_breakdown[key] = energy_breakdown.get(key, 0) + np.sum(energy_taken)
                
                # 2. Add energy to main signals (main and second)
                # First main scale (index 0)
                idx_main = get_idx(main_indices[0], f_idx, p_idx)
                c_main = c_new[:, idx_main]
                # c_main_new = sqrt(c_main^2 + amplify * energy_total)
                # Note: preserve original sign
                energy_main = m_amplify * energy_to_transfer
                c_new[:, idx_main] = np.sign(c_main) * np.sqrt(c_main**2 + energy_main)
                
                # Second main scale (index 1)
                idx_second = get_idx(main_indices[1], f_idx, p_idx)
                c_second = c_new[:, idx_second]
                # c_second_new = sqrt(c_second^2 + (1 - amplify) * energy_total)
                energy_second = (1.0 - m_amplify) * energy_to_transfer
                c_new[:, idx_second] = np.sign(c_second) * np.sqrt(c_second**2 + energy_second)
                
                total_energy_transferred += np.sum(energy_to_transfer)

        # Universal noise floor cutoff
        c_new[c_new**2 < self.eps_thresh] = 0
            
        return c_new, total_energy_transferred, energy_breakdown

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
        c_b, r, stats = self._correct_coefficients(coeffs)
        
        # Reconstruction (block_size x N_filters * N_filters x block_size -> block_size x block_size)
        block = (c_b @ self.basis_b) #/ self.reconstruction_scale
        
        return block, r, stats
