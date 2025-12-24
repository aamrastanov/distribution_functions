import os
import numpy as np
from skimage import io, color, exposure, restoration
from skimage.filters import unsharp_mask
from skimage import img_as_ubyte

# Пути к файлам
base_path = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(base_path, 'input.png')
output_path = os.path.join(base_path, 'output_retinex_simple.png')

if not os.path.exists(input_path):
    print(f"Ошибка: Файл {input_path} не найден!")
    exit(1)

def multi_scale_retinex(img, sigmas=[15, 80, 250]):
    """Классический Multi-Scale Retinex."""
    img = img.astype(float) + 1.0
    retinex = np.zeros_like(img)
    
    from scipy.ndimage import gaussian_filter
    for sigma in sigmas:
        # Логика Retinex: Log(I) - Log(I * G)
        if img.ndim == 3:
            for i in range(3):
                blur = gaussian_filter(img[:,:,i], sigma=sigma)
                retinex[:,:,i] += np.log10(img[:,:,i]) - np.log10(blur + 1.0)
        else:
            blur = gaussian_filter(img, sigma=sigma)
            retinex += np.log10(img) - np.log10(blur + 1.0)
            
    retinex /= len(sigmas)
    
    # Нормализация
    if img.ndim == 3:
        for i in range(3):
            c = retinex[:,:,i]
            retinex[:,:,i] = (c - c.min()) / (c.max() - c.min())
    else:
        retinex = (retinex - retinex.min()) / (retinex.max() - retinex.min())
            
    return np.clip(retinex, 0, 1)

# 1. Загрузка
img_float = io.imread(input_path).astype(float) / 255.0
if img_float.shape[-1] == 4:
    img_float = img_float[:, :, :3]

# 2. Применяем Multi-Scale Retinex
print("Применяю классический Multi-Scale Retinex...")
result = multi_scale_retinex(img_float)

# 3. Финальная "проявка" через адаптивный контраст (CLAHE)
if result.ndim == 3:
    img_lab = color.rgb2lab(result)
    img_lab[:, :, 0] = exposure.equalize_adapthist(img_lab[:, :, 0] / 100.0) * 100.0
    result = color.lab2rgb(img_lab)
else:
    result = exposure.equalize_adapthist(result)

# 4. Легкая резкость
result = unsharp_mask(result, radius=1.5, amount=1.0, channel_axis=-1 if result.ndim==3 else None)

# 5. Сохранение
io.imsave(output_path, img_as_ubyte(np.clip(result, 0, 1)))
print(f"Готово! Результат сохранен в {output_path}")
