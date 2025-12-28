import numpy as np

def get_gaussian_kernel_and_k_crit(sigma):
    # 1. Определяем радиус (3 сигма)
    R = int(np.ceil(3 * sigma))
    
    # 2. Создаем сетку координат m, n
    m = np.arange(-R, R + 1)
    n = np.arange(-R, R + 1)
    M, N = np.meshgrid(m, n)
    
    # 3. Считаем веса Гаусса (не нормированные)
    # g = exp(-(m^2 + n^2) / (2 * sigma^2))
    kernel = np.exp(-(M**2 + N**2) / (2 * sigma**2))
    
    # 4. Нормировка: сумма всех весов (включая центр) должна быть 1
    kernel /= np.sum(kernel)
    
    # 5. Расчет K_crit
    # Взрыв происходит при kx=pi, ky=pi. 
    # Сумма в H: sum( g_mn * (1 - cos(m*pi + n*pi)) )
    # cos(m*pi + n*pi) = (-1)^(m+n)
    
    checkerboard = (-1.0)**(M + N)
    instability_sum = np.sum(kernel * (1 - checkerboard))
    
    k_crit = 1.0 / instability_sum
    
    return kernel, k_crit, M, N

def get_H_value(kx, ky, kernel, K, M, N):
    # M и N - матрицы координат из примера выше
    # kernel - нормированная матрица Гаусса
    
    # Считаем сумму: sum( g_mn * (1 - cos(m*kx + n*ky)) )
    sum_term = np.sum(kernel * (1 - np.cos(M*kx + N*ky)))
    
    return 1 - K * sum_term

# Пример для сигма = 1.0
sigma_val = 1.0
kernel, k_crit, M, N = get_gaussian_kernel_and_k_crit(sigma_val)

print(f"Для сигма = {sigma_val}:")
print(f"Радиус охвата: {M.max()} узлов")
print(f"Критическое K: {k_crit:.4f}")