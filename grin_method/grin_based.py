
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.fft import fft2, ifft2, fftfreq, fftshift
from mpl_toolkits.mplot3d import Axes3D # Для 3D графики

import sys
import os

# Добавляем путь для импорта, если скрипт запущен напрямую
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from base_methods import get_gaussian_kernel_and_k_crit

# --- ПАРАМЕТРЫ ---
N = 100
sigma_blur_real = 5.0  # Какое размытие мы накладываем на данные
sigma_model = 5.0      # Какое размытие (радиус взаимодействия) мы закладываем в модель восстановления
K_factor = 0.9        # Доля от K_crit (0.9..0.999) для стабильности
num_points = 20         # Количество случайных сигналов

# --- 1. ГЕНЕРАЦИЯ ДАННЫХ ---
np.random.seed(42) # Для воспроизводимости
A = np.zeros((N, N))

# Генерируем случайные точки
for _ in range(num_points):
    ry = np.random.randint(10, N-10)
    rx = np.random.randint(10, N-10)
    A[ry, rx] = 1.0

# Используем mode='wrap' для чистоты FFT эксперимента
B = gaussian_filter(A, sigma=sigma_blur_real, mode='constant', cval=0.0)

# --- 2. ПОДГОТОВКА GRIN KERNEL И H ---
# Получаем локальное ядро взаимодействия (гауссовы веса соседей)
kernel_g, k_crit, M, N_grid = get_gaussian_kernel_and_k_crit(sigma_model)

print(f"Sigma Model: {sigma_model}")
print(f"Kernel Size: {kernel_g.shape}")
print(f"K Critical: {k_crit:.6f}")

K = k_crit * K_factor
print(f"Using K: {K:.6f} (Factor: {K_factor})")

# Строим сетку частот
# fftfreq возвращает [0, 1, ..., N/2-1, -N/2, ..., -1] / N
kx_1d = fftfreq(N) * 2 * np.pi
ky_1d = fftfreq(N) * 2 * np.pi
KX, KY = np.meshgrid(kx_1d, ky_1d)

# ВЫЧИСЛЕНИЕ H(kx, ky)
# H = 1 - K * sum( g_mn * (1 - cos(m*kx + n*ky)) )
# Нам нужно векторизовать это вычисление по всему гриду частот (KX, KY)
# M, N_grid - это координаты соседей (shape: R x R)
# Kernel_g - веса соседей (shape: R x R)

# Расширяем размерности для бродкастинга:
# Grid (neighbors): (R, R, 1, 1)
# Frequencies:      (1, 1, H, W)

M_exp = M[:, :, np.newaxis, np.newaxis]
N_exp = N_grid[:, :, np.newaxis, np.newaxis]
G_exp = kernel_g[:, :, np.newaxis, np.newaxis]

KX_exp = KX[np.newaxis, np.newaxis, :, :]
KY_exp = KY[np.newaxis, np.newaxis, :, :]

# Считаем аргумент косинуса для каждой пары (сосед, частота)
# phase = m*kx + n*ky
phase = M_exp * KX_exp + N_exp * KY_exp

# Считаем сумму по соседям
# sum_term = sum( g_mn * (1 - cos(phase)) )
sum_term = np.sum(G_exp * (1 - np.cos(phase)), axis=(0, 1))

H = 1 - K * sum_term

# --- 3. ВОССТАНОВЛЕНИЕ ---
# F_restored = iFFT( FFT(B) / H )
F_B = fft2(B)

# Обработка деления на ноль (хотя при K < K_crit H не должно быть 0, кроме k=0)
# Но при k=0: sum_term = 0 -> H = 1. Все ок.
# Однако, H может быть очень малым на высоких частотах.
epsilon = 1e-10
F_Restored_Spectr = F_B / (H + epsilon)

Restored = np.real(ifft2(F_Restored_Spectr))
Restored = np.maximum(Restored, 0) # Убираем отрицательные выбросы (физическое ограничение)

# --- 4. ВИЗУАЛИЗАЦИЯ ---
# Нормировка для красивого вывода (хотя Grin метод сохраняет энергию физически)
# Проверим сохранение энергии
energy_in = np.sum(B)
energy_out = np.sum(Restored)
print(f"Energy Input: {energy_in:.2f}")
print(f"Energy Output: {energy_out:.2f}")

Restored_Norm = (Restored - Restored.min()) / (Restored.max() - Restored.min())
B_Norm = (B - B.min()) / (B.max() - B.min())

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original
ax = axes[0]
im = ax.imshow(A, cmap='gray')
ax.set_title("Original (Point)")
plt.colorbar(im, ax=ax)

# Blurred
ax = axes[1]
im = ax.imshow(B, cmap='gray')
ax.set_title(f"Blurred (Sigma={sigma_blur_real})")
plt.colorbar(im, ax=ax)

# Restored
ax = axes[2]
im = ax.imshow(Restored, cmap='hot')
ax.set_title(f"Restored (Grin K={K:.4f})")
plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.show()

# 3D Interactive Plot - Triple Comparison
fig = plt.figure(figsize=(24, 8))
X_grid, Y_grid = np.meshgrid(np.arange(N), np.arange(N))

# 1. Original (3D)
ax1 = fig.add_subplot(131, projection='3d')
surf1 = ax1.plot_surface(X_grid, Y_grid, A, cmap='gray', linewidth=0, antialiased=False)
ax1.set_title("Original (Point Sources)")
ax1.set_zlabel('Intensity')

# 2. Blurred (3D)
ax2 = fig.add_subplot(132, projection='3d')
surf2 = ax2.plot_surface(X_grid, Y_grid, B, cmap='gray', linewidth=0, antialiased=False)
ax2.set_title(f"Blurred (Sigma={sigma_blur_real})")
ax2.set_zlabel('Intensity')

# 3. Restored (3D)
ax3 = fig.add_subplot(133, projection='3d')
surf3 = ax3.plot_surface(X_grid, Y_grid, Restored, cmap='hot', linewidth=0, antialiased=False)
ax3.set_title(f"Restored (Grin K={K:.4f})")
ax3.set_zlabel('Intensity')

plt.tight_layout()
plt.show()
