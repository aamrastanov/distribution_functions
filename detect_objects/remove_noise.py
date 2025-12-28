import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, restoration, img_as_float, filters, exposure

# 1. Загрузка
base_path = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(base_path, 'input_small_part.png')
image = img_as_float(io.imread(input_path, as_gray=True))

# --- ШАГ 1: ЭКСТРЕМАЛЬНОЕ УСИЛЕНИЕ КОНТРАСТА ---
# Вытаскиваем структуру из-под шума ДО фильтрации.
# clip_limit: чем выше, тем агрессивнее усиление (попробуйте 0.05 - 0.1)
image_enhanced = exposure.equalize_adapthist(image, clip_limit=0.1)

# --- ШАГ 2: ОЧЕНЬ МЯГКИЙ TV-ФИЛЬТР ---
# Почти ювелирная очистка, чтобы не потерять слабые тени
image_tv = restoration.denoise_tv_chambolle(image_enhanced, weight=0.02)

# --- ШАГ 3: ДЕЛИКАТНЫЕ ВЕЙВЛЕТЫ ---
# Уменьшаем количество уровней, чтобы не превратить всё в кашу
image_wavelet = restoration.denoise_wavelet(image_tv, 
                                            method='VisuShrink', 
                                            mode='soft', 
                                            wavelet_levels=1, 
                                            wavelet='db2')

# --- ШАГ 4: ГРАДИЕНТ (ГРАНИЦЫ) ---
edges = filters.scharr(image_wavelet)

# Визуализация
fig, ax = plt.subplots(1, 4, figsize=(20, 5))
ax[0].imshow(image, cmap='gray'); ax[0].set_title('1. Оригинал')
ax[1].imshow(image_enhanced, cmap='gray'); ax[1].set_title('2. Усиление (CLAHE)')
ax[2].imshow(image_wavelet, cmap='gray'); ax[2].set_title('2. Очищенная структура')
ax[3].imshow(edges, cmap='magma'); ax[3].set_title('3. Границы')

for a in ax: a.axis('off')
plt.tight_layout()
plt.show()