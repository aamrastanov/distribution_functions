import os
import numpy as np
from skimage import io, color, restoration


# Получаем путь к папке, где лежит сам скрипт
base_path = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(base_path, 'input.png')
output_path = os.path.join(base_path, 'output.png')

# Загрузка вашего изображения
if not os.path.exists(input_path):
    print(f"Ошибка: Файл {input_path} не найден!")
    exit(1)

# --- Продвинутый Multi-Scale Retinex с восстановлением цвета (MSRCR) ---

def msr_with_color_restoration(img, sigmas=[15, 80, 250], alpha=125.0, beta=46.0, G=192.0, b=-30.0):
    """
    MSRCR: Добавляет восстановление цвета и более сложную нормализацию.
    """
    img = img.astype(float) + 1.0
    
    # 1. Вычисляем Multi-Scale Retinex для каждого канала
    retinex = np.zeros_like(img)
    from scipy.ndimage import gaussian_filter
    for sigma in sigmas:
        for i in range(img.shape[2] if img.ndim==3 else 1):
            if img.ndim == 3:
                blur = gaussian_filter(img[:,:,i], sigma=sigma)
                retinex[:,:,i] += np.log10(img[:,:,i]) - np.log10(blur + 1.0)
            else:
                blur = gaussian_filter(img, sigma=sigma)
                retinex += np.log10(img) - np.log10(blur + 1.0)
    retinex /= len(sigmas)
    
    # 2. Восстановление цвета (Color Restoration Factor)
    if img.ndim == 3:
        img_sum = np.sum(img, axis=2, keepdims=True)
        color_restoration = beta * (np.log10(alpha * img) - np.log10(img_sum))
        result = G * (retinex * color_restoration + b)
    else:
        result = G * (retinex + b)
    
    # Нормализация (продвинутый вариант с обрезкой выбросов)
    if result.ndim == 3:
        for i in range(3):
            low = np.percentile(result[:,:,i], 1)
            high = np.percentile(result[:,:,i], 99)
            result[:,:,i] = np.clip((result[:,:,i] - low) / (high - low), 0, 1)
    else:
        low = np.percentile(result, 1)
        high = np.percentile(result, 99)
        result = np.clip((result - low) / (high - low), 0, 1)
        
    return result

# 1. Загрузка
img_float = io.imread(input_path).astype(float) / 255.0
if img_float.shape[-1] == 4:
    img_float = img_float[:, :, :3]

# 2. Основная обработка: MSRCR
print("Применяю MSRCR (это может занять несколько секунд)...")
result = msr_with_color_restoration(img_float)

# 3. Адаптивное шумоподавление (NL-Means)
# Убирает "зернистость", которую мог усилить Retinex
print("Убираю шумы...")
from skimage.restoration import denoise_nl_means, estimate_sigma
sigma_est = np.mean(estimate_sigma(result, channel_axis=-1 if result.ndim==3 else None))
result = denoise_nl_means(result, h=1.15 * sigma_est, fast_mode=True, 
                           patch_size=5, patch_distance=3, channel_axis=-1 if result.ndim==3 else None)

# 4. Финальный контраст и резкость
from skimage import exposure, color
if result.ndim == 3:
    img_lab = color.rgb2lab(result)
    img_lab[:, :, 0] = exposure.equalize_adapthist(img_lab[:, :, 0] / 100.0) * 100.0
    result = color.lab2rgb(img_lab)
else:
    result = exposure.equalize_adapthist(result)

from skimage.filters import unsharp_mask
result = unsharp_mask(result, radius=1.0, amount=1.5, channel_axis=-1 if result.ndim==3 else None)

from skimage import img_as_ubyte
io.imsave(output_path, img_as_ubyte(np.clip(result, 0, 1)))
print(f"Готово! Шедевр сохранен в {output_path}")