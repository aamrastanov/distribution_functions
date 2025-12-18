import numpy as np
import logging

def G(x, mu, sigma):
    """Базовая Гауссиана с жесткой отсечкой (интервалы строгого нуля)."""
    g = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    # Те самые интервалы строгого нуля, о которых говорилось в документе
    g[np.abs(x - mu) > 3 * sigma] = 0.0
    return g

def generate_f_n(x, shift_index, k, step, sigma, num_blocks):
    """
    Генерирует одну базисную функцию F_n(x) со сдвигом shift_index.
    Это прямая реализация вашего кода.
    """
    # Список центров mu_i
    mu = [i * step for i in range(1, k * (num_blocks + 1) + 2 + shift_index)]
    
    f_n = np.zeros_like(x)
    for b in range(num_blocks):
        block = np.ones_like(x)
        for m in range(k):
            # Внутри блока перемножаем k смещенных Гауссиан
            block *= G(x, mu[b * k + m + shift_index], sigma)
        # Знакочередование блоков
        f_n += (-1) ** b * block
    return f_n

def get_wgm_basis_matrix(kernel_size, k, step, num_blocks):
    """
    Создает набор базисных функций (ядер) для обработки изображения.
    """
    sigma = k / 3.0
    x = np.linspace(0, kernel_size, kernel_size)
    
    # Мы генерируем несколько вариантов сдвигов (F1, F2, F3...)
    # для детекции разных фаз текстуры дна
    basis_kernels = []
    for s in range(k + 1): # Генерируем сдвиги от 0 до k
        f_n = generate_f_n(x, shift_index=s, k=k, step=step, sigma=sigma, num_blocks=num_blocks)
        # Превращаем 1D функцию в 2D ядро (внешнее произведение)
        kernel_2d = np.outer(f_n, f_n)
        # Нормировка для сохранения энергии сигнала
        if np.max(kernel_2d) > 0:
            kernel_2d /= np.max(kernel_2d)
        basis_kernels.append(kernel_2d)
        
    return basis_kernels