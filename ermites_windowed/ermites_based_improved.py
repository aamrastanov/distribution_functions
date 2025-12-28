import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.special import eval_hermite
from scipy.fft import fft2, ifft2
import sys
import os

# Добавляем корневую библиотеку в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base.gradient_ops import poisson_solve, compute_gradients

# --- ПАРАМЕТРЫ ---
N = 100
sigma_blur = 3.0
window_size = 51
step = 10
n_max = 5  # Порядок полиномов Эрмита
sigma_large = 3.0
sigma_small = 0.3
alpha = 0.8  # Коэффициент подавления большой сигмы
beta_max = 5.0 # Ограничитель усиления
noise_threshold = 1e-5 # Порог шума for skipping empty windows

def get_hermite_basis(size, sigma, n_orders):
    """Генерирует ортонормированный базис функций Эрмита"""
    x = np.linspace(-size/2, size/2, size)
    basis = []
    for n in range(n_orders):
        # Используем "физические" полиномы Эрмита (H_n) для ортогональности с весом e^(-x^2)
        h = eval_hermite(n, x / sigma)
        phi = h * np.exp(- (x**2) / (2 * sigma**2))
        # Ортонормировка
        norm = np.sqrt(np.sum(phi**2))
        if norm > 1e-9:
            phi /= norm
        basis.append(phi)
    return np.array(basis)

def process_gradient_1d(G_line, basis_l, basis_s, alpha, beta_limit, noise_thresh):
    """Обработка одной линии градиента (1D) по твоему алгоритму"""
    out = G_line.copy()
    L = len(G_line)
    W = window_size
    
    # Скользящее окно
    for start in range(0, L - W + 1, step):
        end = start + W
        segment = out[start:end]
        
        # Проверка "ядра" окна (центральная часть размером step)
        # Если там нет сигнала выше шума - пропускаем обработку
        core_start = (W - step) // 2
        core_end = core_start + step
        core = segment[core_start:core_end]
        if np.max(np.abs(core)) < noise_thresh:
            continue
        
        # 1. Разложение по большой сигме
        d = [np.dot(segment, basis_l[n]) for n in range(n_max)]
        
        # Энергия большой сигмы (включая d0)
        e_large = sum(val**2 for val in d)
        
        # Подавляем большую сигму (включая d0)
        d_new = [val * alpha for val in d]
        
        # 2. Вычисляем остаток
        reconstruct_l = np.zeros(W)
        for n in range(n_max):
            reconstruct_l += d[n] * basis_l[n]
        residue = segment - reconstruct_l
        
        # 3. Разложение остатка по малой сигме
        c = [np.dot(residue, basis_s[n]) for n in range(n_max)]
        e_small = sum(val**2 for val in c[1:])
        
        # 4. Перекачка энергии
        delta_e = (1 - alpha**2) * e_large
        beta = np.sqrt(1 + delta_e / (e_small + 1e-6))
        beta = min(beta, beta_limit)
        
        # Усиливаем малую сигму (включая c0)
        c_new = [val * beta for val in c]
        
        # 5. Реконструкция сегмента
        new_segment = np.zeros(W)
        for n in range(n_max):
            new_segment += d_new[n] * basis_l[n]
            new_segment += c_new[n] * basis_s[n]
            
        # Записываем обратно (рекурсивное обновление)
        out[start:end] = new_segment
        
    return out

# --- 1. ПОДГОТОВКА ДАННЫХ ---
A = np.zeros((N, N))
A[25, 25] = 1.0  # Точечный сигнал

B = gaussian_filter(A, sigma=sigma_blur)

# --- 2. ГРАДИЕНТЫ ---
# Используем compute_gradients для согласованности с poisson_solve
# (Forward Difference, как и ожидает решатель)
DBx, DBy = compute_gradients(B)

# --- 3. ПРИМЕНЕНИЕ АЛГОРИТМА ---
print("Лог: Генерация базисов...")
basis_l = get_hermite_basis(window_size, sigma_large, n_max)
basis_s = get_hermite_basis(window_size, sigma_small, n_max)

print("Лог: Обработка градиента X (построчно)...")
Cx_grad = np.zeros_like(DBx)
for i in range(N):
    Cx_grad[i, :] = process_gradient_1d(DBx[i, :], basis_l, basis_s, alpha, beta_max, noise_threshold)

print("Лог: Обработка градиента Y (постолбцово)...")
Cy_grad = np.zeros_like(DBy)
for j in range(N):
    Cy_grad[:, j] = process_gradient_1d(DBy[:, j], basis_l, basis_s, alpha, beta_max, noise_threshold)

# --- 4. РЕШЕНИЕ УРАВНЕНИЯ ПУАССОНА (Через базовую библиотеку) ---
print("Лог: Восстановление изображения (Пуассон)...")
C = poisson_solve(Cx_grad, Cy_grad)

# --- 5. ВЫРАВНИВАНИЕ ЯРКОСТИ ---
C = (C - np.min(C)) / (np.max(C) - np.min(C)) # Нормализация 0..1
C *= np.sum(B) / np.sum(C) # Выравнивание по суммарной энергии с B

# --- 7. ВИЗУАЛИЗАЦИЯ ---
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
titles = ['A (Original)', 'B (Blurred)', 'DBx (Grad X)', 'DBy (Grad Y)', 'C (Recovered)']
images = [A, B, DBx, DBy, C]

for ax, img, title in zip(axes, images, titles):
    im = ax.imshow(img, cmap='hot' if 'Grad' in title else 'gray')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()