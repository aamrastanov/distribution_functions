
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, img_as_float, img_as_ubyte
from skimage.restoration import denoise_tv_chambolle
from scipy.fft import fft2, ifft2, fftfreq

import sys
import os

# Добавляем путь для импорта base_methods
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from base_methods import get_gaussian_kernel_and_k_crit

# --- ПАРАМЕТРЫ ---
INPUT_IMAGE = 'akvarium.png'
OUTPUT_IMAGE = 'akvarium_grin_restored.png'

# Параметры модели (требуют настройки под конкретное размытие на фото)
sigma_model = 40.0      # Радиус взаимодействия (насколько "широко" растекся сигнал)
K_factor = 0.9        # Сила восстановления (0.9..0.99)
denoise_weight = 0.02  # Сила предварительного шумоподавления (0.01..0.2)
epsilon = 1e-6         # Стабилизатор деления

# --- 1. ЗАГРУЗКА И ПОДГОТОВКА ---
input_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), INPUT_IMAGE)
if not os.path.exists(input_path):
    print(f"Ошибка: Файл {input_path} не найден!")
    sys.exit(1)

# Читаем и переводим в grayscale float [0, 1]
img = io.imread(input_path)
if img.ndim == 3:
    # Если есть альфа-канал (RGBA), берем только RGB
    if img.shape[2] == 4:
        img = img[:, :, :3]
    gray = color.rgb2gray(img)
else:
    gray = img_as_float(img)

# Активное шумоподавление перед деконволюцией (Total Variation)
if denoise_weight > 0:
    print(f"Лог: Шумоподавление (TV, weight={denoise_weight})...")
    gray = denoise_tv_chambolle(gray, weight=denoise_weight)

H_img, W_img = gray.shape
print(f"Размер изображения: {W_img}x{H_img}")

# --- 2. ПОДГОТОВКА ГРИН-ФИЛЬТРА (Оптимизировано по памяти) ---
kernel_g, k_crit, M, N_grid = get_gaussian_kernel_and_k_crit(sigma_model)
K = k_crit * K_factor

print(f"K Critical: {k_crit:.6f}, Using K: {K:.6f}")

# Вычисляем H через FFT от ядра (это эквивалентно сумме косинусов)
# Это гораздо быстрее и не требует 26ГБ памяти
print("Лог: Подготовка передаточной функции H...")

# 1. Создаем пустое ядро размером с картинку
padded_kernel = np.zeros((H_img, W_img))
kh, kw = kernel_g.shape
# Вставляем маленькое ядро в центр
start_h = (H_img - kh) // 2
start_w = (W_img - kw) // 2
padded_kernel[start_h:start_h+kh, start_w:start_w+kw] = kernel_g

# 2. Сдвигаем ядро так, чтобы центр был в (0,0) для корректного FFT
padded_kernel = np.fft.ifftshift(padded_kernel)

# 3. FFT от ядра дает нам сумму (g_mn * exp(i*k*r))
# Так как ядро симметрично, мнимая часть будет нулевой (сумма косинусов)
kernel_fft = np.real(fft2(padded_kernel))

# 4. H = 1 - K * (Sum(g) - Sum(g*cos)) = 1 - K * (1 - kernel_fft)
# так как Sum(g) = 1 (ядро нормировано)
H_total = 1 - K * (1 - kernel_fft)

print(f"Диапазон фильтра H: [{H_total.min():.4f}, {H_total.max():.4f}]")
if H_total.min() < 0.05:
    print("Внимание: Слишком низкое значение H, возможен взрыв шума!")

# --- 3. ВОССТАНОВЛЕНИЕ ---
print("Лог: Обработка спектра (FFT)...")
F_gray = fft2(gray)
# Деление в спектре
F_restored = F_gray / (H_total + epsilon)

# Обратно в пространство
restored = np.real(ifft2(F_restored))

# --- 4. ПОСТ-ОБРАБОТКА ---
# Обрезаем нефизичные значения
restored = np.clip(restored, 0, 1)

# Выравниваем яркость (по среднему)
mean_orig = np.mean(gray)
mean_restored = np.mean(restored)
if mean_restored > 0:
    restored *= (mean_orig / mean_restored)
restored = np.clip(restored, 0, 1)

# --- 5. СОХРАНЕНИЕ И ВИЗУАЛИЗАЦИЯ ---
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_IMAGE)
io.imsave(output_path, img_as_ubyte(restored))
print(f"Результат сохранен в: {output_path}")

# Вывод сравнения
fig, axes = plt.subplots(1, 3, figsize=(24, 8))
axes[0].imshow(gray, cmap='gray')
axes[0].set_title("Original (Denoised)")
axes[0].axis('off')

# Визуализация передаточной функции H (ЗУМ в центр спектра)
H_shifted = np.fft.fftshift(H_total)
cy, cx = H_img // 2, W_img // 2
zoom = 50 # Покажем 100x100 пикселей центра
H_zoom = H_shifted[cy-zoom:cy+zoom, cx-zoom:cx+zoom]

im_h = axes[1].imshow(H_zoom, cmap='viridis')
axes[1].set_title(f"H-Filter Zoom (center 100x100)\nSigma={sigma_model}")
fig.colorbar(im_h, ax=axes[1], shrink=0.6)

axes[2].imshow(restored, cmap='gray')
axes[2].set_title(f"Restored (K_f={K_factor})")
axes[2].axis('off')

plt.tight_layout()
plt.show()
