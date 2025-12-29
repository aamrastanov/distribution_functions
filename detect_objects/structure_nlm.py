import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float, restoration, exposure

# 1. Твоя загрузка
base_path = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(base_path, 'input_small_part.png')
image = img_as_float(io.imread(input_path, as_gray=True))

# 2. Оценка шума (автоматически)
sigma_est = np.mean(restoration.estimate_sigma(image))

# 3. Метод Нелокальных Средних (NLM)
# Это "тяжелая артиллерия". Он ищет структуру, сравнивая фрагменты по всей картинке.
# h: коэффициент фильтрации (чем выше, тем сильнее "гладит")
image_nlm = restoration.denoise_nl_means(image, h=1.15 * sigma_est, fast_mode=True,
                                        patch_size=10, patch_distance=17)

# 4. Усиление того, что выжило (Адаптивный контраст)
# После NLM структура станет видна, но будет бледной. Вытягиваем её.
final_structure = exposure.equalize_adapthist(image_nlm, clip_limit=0.05)

# Визуализация
fig, ax = plt.subplots(1, 3, figsize=(18, 6))
ax[0].imshow(image, cmap='gray'); ax[0].set_title('1. Оригинал')
ax[1].imshow(image_nlm, cmap='gray'); ax[1].set_title('2. Структурный NLM')
ax[2].imshow(final_structure, cmap='magma'); ax[2].set_title('3. Вытянутый объект')

for a in ax: a.axis('off')
plt.show()