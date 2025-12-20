
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import logging
import sys
import os

# Добавляем путь к родительской директории для импорта modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from basis_engine import get_wgm_basis_matrix

def main():
    logging.basicConfig(level=logging.INFO)

    # Параметры из processing_steps.py
    k = 3
    step = 1.0
    num_blocks = 16
    sigma = 1

    print(f"Generating Discrete WGM basis functions with: k={k}, step={step}, blocks={num_blocks}, sigma={sigma}")
    
    # 1. Используем get_wgm_basis_matrix (дискретные ядра)
    kernels = get_wgm_basis_matrix(sigma=sigma, k=k, step=step, num_blocks=num_blocks)
    
    num_kernels = len(kernels)
    # Размер определяется внутри get_wgm_basis_matrix динамически
    kernel_size = kernels[0].shape[0]
    print(f"Kernel discrete grid size: {kernel_size}x{kernel_size}")

    # Создаем дискретную сетку координат
    x = np.arange(kernel_size)
    y = np.arange(kernel_size)
    X, Y = np.meshgrid(x, y)

    # Создаем фигуру с 3D подграфиками
    fig = plt.figure(figsize=(5 * num_kernels, 5))
    
    for i, kernel in enumerate(kernels):
        # Добавляем 3D subplot (1 строка, num_kernels столбцов, индекс i+1)
        ax = fig.add_subplot(1, num_kernels, i+1, projection='3d')
        
        # Строим поверхность
        # Используем rcount/ccount = kernel_size, чтобы показать каждый пиксель
        surf = ax.plot_surface(X, Y, kernel, cmap=cm.viridis,
                               linewidth=0.5, edgecolors='k', antialiased=False, # Добавил edgecolors чтобы видеть сетку
                               rcount=kernel_size, ccount=kernel_size)
        
        ax.set_title(f'Discrete Kernel #{i}\nShift index: {i}')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.set_zlabel('Amplitude')
        
        # Настройка начального угла обзора
        ax.view_init(elev=30, azim=45)

    plt.suptitle(f'Discrete 3D Visualization (Actual Kernels)\nsize={kernel_size}x{kernel_size}, k={k}, blocks={num_blocks}, sigma={sigma}')
    
    print("Displaying discrete 3D plot...")
    plt.show()

if __name__ == "__main__":
    main()
