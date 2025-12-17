import numpy as np
import logging

def create_gaussian_block(size, sigma):
    """Создает элементарный гауссов блок (B_j)."""
    ax = np.linspace(-(size // 2), size // 2, size)
    gauss = np.exp(-0.5 * (np.square(ax) / np.square(sigma)))
    return gauss / np.sum(gauss)

def generate_wgm_kernel(size, k_blocks, sigma):
    """
    Генерирует знакочередующийся поезд гауссовых сгустков (F(x)).
    size: размер ядра
    k_blocks: количество блоков в поезде
    """
    logging.info(f"Генерация ядра ВГМБ: размер {size}, блоков {k_blocks}")
    kernel_1d = np.zeros(size)
    block_size = size // k_blocks
    
    for j in range(k_blocks):
        block = create_gaussian_block(block_size, sigma)
        start = j * block_size
        sign = (-1) ** j
        kernel_1d[start:start+block_size] = sign * block
        
    # Создаем 2D ядро путем внешнего произведения
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    return kernel_2d