
import matplotlib.pyplot as plt
import numpy as np
import logging
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from basis_engine import get_wgm_basis_matrix

def main():
    logging.basicConfig(level=logging.INFO)

    # Параметры из processing_steps.py
    k = 3
    step = 1.0
    num_blocks = 6
    sigma = 1

    print(f"Generating WGM basis functions with: k={k}, step={step}, blocks={num_blocks}, sigma={sigma}")
    
    # Получение 2D ядер
    kernels = get_wgm_basis_matrix(sigma=sigma, k=k, step=step, num_blocks=num_blocks)
    
    plt.figure(figsize=(10, 6))
    
    x_axis = np.arange(22)
    
    # Берем срез при y=0 (индекс 0) для каждого ядра
    for i, kernel in enumerate(kernels):
        # Строго следуя запросу: срез при y=0
        profile_1d = kernel[10, :]
        
        plt.plot(x_axis, profile_1d, label=f'Kernel #{i} (Shift={i})')

    plt.title(f'1D Profiles at y=0\nk={k}, blocks={num_blocks}, step={step}')
    plt.xlabel('x')
    plt.ylabel('Amplitude')
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    print("Displaying 1D plot...")
    plt.show()

if __name__ == "__main__":
    main()
