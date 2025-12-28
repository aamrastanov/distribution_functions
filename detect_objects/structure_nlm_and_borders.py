import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float, restoration, exposure, filters

# 1. Загрузка (твоя логика путей)
base_path = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(base_path, 'input_small_part.png')
image = img_as_float(io.imread(input_path, as_gray=True))

# 2. Автоматическая оценка шума
# Это важно: NLM работает лучше всего, когда знает точный уровень "хаоса"
sigma_est = np.mean(restoration.estimate_sigma(image))
print(f"Estimated noise level (sigma): {sigma_est:.4f}")

# 3. ОСНОВНОЙ ЭТАП: Non-Local Means (NLM)
# Ищем структуру, усредняя похожие паттерны по всему фрагменту
image_nlm = restoration.denoise_nl_means(
    image, 
    h=1.2 * sigma_est,    # Коэффициент "агрессивности" фильтра
    fast_mode=True,       # Оптимизация по скорости
    patch_size=5,         # Размер окна сравнения (структура)
    patch_distance=6      # Радиус поиска похожих патчей
)

# 4. УСИЛЕНИЕ СКРЫТЫХ ДЕТАЛЕЙ (CLAHE)
# Вытягиваем яркость найденных структур
image_clahe = exposure.equalize_adapthist(image_nlm, clip_limit=0.03)

# 5. ФИНАЛЬНЫЙ КОНТУР (Unsharp Masking)
# Подсвечиваем границы, которые NLM смог вытащить из-под шума
image_final = filters.unsharp_mask(image_clahe, radius=1.0, amount=1.5)

# 6. Визуализация всего процесса
fig, ax = plt.subplots(1, 4, figsize=(24, 6))

ax[0].imshow(image, cmap='gray')
ax[0].set_title('1. Оригинал (Шум)')

ax[1].imshow(image_nlm, cmap='gray')
ax[1].set_title('2. NLM (Выделение структуры)')

ax[2].imshow(image_clahe, cmap='gray')
ax[2].set_title('3. CLAHE (Усиление контраста)')

ax[3].imshow(image_final, cmap='magma') # Magma лучше проявляет слабые границы
ax[3].set_title('4. Итог (Контуры объектов)')

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()

# Опционально: сохранение результата
# io.imsave(os.path.join(base_path, 'result_structure.png'), (image_final * 255).astype(np.uint8))