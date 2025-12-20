
import matplotlib.pyplot as plt
import numpy as np
import logging
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from basis_engine import get_wgm_basis_matrix

def main():
    # Настройка логирования, чтобы видеть процесс
    logging.basicConfig(level=logging.INFO)

    # Параметры точно как в processing_steps.py (строка 107 и рядом)
    # k=3, num_blocks=6, step=1 -> эффективная ширина ~ 22. 25 с небольшим запасом.
    k = 3
    step = 1.0
    num_blocks = 6
    sigma = 1

    print(f"Generating WGM basis functions with: k={k}, step={step}, blocks={num_blocks}, sigma={sigma}")
    
    # Получение базисных ядер
    kernels = get_wgm_basis_matrix(sigma=sigma, k=k, step=step, num_blocks=num_blocks)
    
    num_kernels = len(kernels)
    print(f"Generated {num_kernels} kernels.")

    # Визуализация
    # Создаем фигуру с подграфиками. Расположим их в одну строку.
    fig, axes = plt.subplots(1, num_kernels, figsize=(4 * num_kernels, 4))
    if num_kernels == 1:
        axes = [axes]
    
    for i, kernel in enumerate(kernels):
        ax = axes[i]
        # Используем 'viridis' или 'plasma' для лучшего контраста, или 'gray'
        im = ax.imshow(kernel, cmap='viridis', interpolation='nearest')
        ax.set_title(f'Kernel #{i}\nShift index: {i}')
        # Добавляем цветовой бар для каждого, чтобы видеть амплитуду
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Отобразим min/max значения
        d_min = kernel.min()
        d_max = kernel.max()
        ax.set_xlabel(f'Min: {d_min:.3f}, Max: {d_max:.3f}')

    plt.suptitle(f'WGM Basis Functions (k={k}, blocks={num_blocks}, step={step})')
    plt.tight_layout()
    
    print("Displaying plot...")
    plt.show()

if __name__ == "__main__":
    main()
