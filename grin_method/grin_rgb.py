
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float, img_as_ubyte
from skimage.restoration import denoise_tv_chambolle
from scipy.fft import fft2, ifft2

import sys
import os

# Добавляем путь для импорта base_methods
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from base_methods import get_gaussian_kernel_and_k_crit

# --- ПАРАМЕТРЫ ПО КАНАЛАМ (R, G, B) ---
sigmas = [50.0, 70.0, 100.0]     # sigma_model для каждого канала
k_factors = [0.9, 0.9, 0.9]   # k_factor для каждого канала
denoise_weights = [0.02, 0.02, 0.02] # шум для каждого канала

INPUT_IMAGE = 'akvarium.png'
OUTPUT_IMAGE = 'akvarium_grin_rgb.png'
epsilon = 1e-6

# --- 1. ЗАГРУЗКА ---
input_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), INPUT_IMAGE)
if not os.path.exists(input_path):
    print(f"Ошибка: Файл {input_path} не найден!")
    sys.exit(1)

img_orig = io.imread(input_path)
if img_orig.shape[2] == 4:
    img_orig = img_orig[:, :, :3]
img_float = img_as_float(img_orig)

H_img, W_img, _ = img_float.shape
print(f"Размер изображения: {W_img}x{H_img}")

# --- 2. ОБРАБОТКА ПО КАНАЛАМ ---
img_restored = np.zeros_like(img_float)
channels = ['Red', 'Green', 'Blue']

for i in range(3):
    print(f"\n--- Обработка канала {channels[i]} ---")
    ch_data = img_float[:, :, i]
    
    # 2.1 Шумоподавление
    if denoise_weights[i] > 0:
        print(f"Лог: Шумоподавление (TV, weight={denoise_weights[i]})...")
        ch_data = denoise_tv_chambolle(ch_data, weight=denoise_weights[i])
    
    # 2.2 Подготовка фильтра H
    kernel_g, k_crit, _, _ = get_gaussian_kernel_and_k_crit(sigmas[i])
    K = k_crit * k_factors[i]
    print(f"Sigma={sigmas[i]}, K_crit={k_crit:.6f}, Using K={K:.6f}")
    
    # Создаем и сдвигаем ядро
    padded_kernel = np.zeros((H_img, W_img))
    kh, kw = kernel_g.shape
    # Если ядро больше картинки, кидаем ошибку как просил пользователь
    start_h = (H_img - kh) // 2
    start_w = (W_img - kw) // 2
    padded_kernel[start_h:start_h+kh, start_w:start_w+kw] = kernel_g
    padded_kernel = np.fft.ifftshift(padded_kernel)
    
    # FFT от ядра
    kernel_fft = np.real(fft2(padded_kernel))
    H_total = 1 - K * (1 - kernel_fft)
    
    # 2.3 Деконволюция
    print("Лог: FFT Деконволюция...")
    F_ch = fft2(ch_data)
    F_restored = F_ch / (H_total + epsilon)
    res_ch = np.real(ifft2(F_restored))
    
    # 2.4 Пост-обработка канала
    res_ch = np.clip(res_ch, 0, 1)
    # Коррекция яркости
    mean_orig = np.mean(img_float[:, :, i])
    mean_res = np.mean(res_ch)
    if mean_res > 0:
        res_ch *= (mean_orig / mean_res)
    
    img_restored[:, :, i] = np.clip(res_ch, 0, 1)

# --- 3. СОХРАНЕНИЕ И ВЫВОД ---
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_IMAGE)
io.imsave(output_path, img_as_ubyte(img_restored))
print(f"\nГотово! Результат сохранен в: {output_path}")

# Визуализация
fig, axes = plt.subplots(1, 2, figsize=(20, 10))
axes[0].imshow(img_orig)
axes[0].set_title("Original RGB")
axes[0].axis('off')

axes[1].imshow(img_restored)
axes[1].set_title("Restored RGB (Grin RGB)")
axes[1].axis('off')

plt.tight_layout()
plt.show()
