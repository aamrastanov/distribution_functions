import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.decomposition import PCA
import umap

# 1. Загрузка фрагмента (твой способ путей)
base_path = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(base_path, 'input_small_part.png')
image = img_as_float(io.imread(input_path, as_gray=True))

h, w = image.shape
patch_size = (5, 5) # Размер окна для анализа структуры

# 2. Извлекаем патчи (превращаем каждый пиксель в вектор контекста)
patches = extract_patches_2d(image, patch_size)
data = patches.reshape(patches.shape[0], -1)

# 3. БЫСТРЫЙ ЭТАП: PCA
# Шум хаотичен, а сигнал (граница) коррелирован. 
# Первая компонента PCA часто вытаскивает именно "суть".
pca = PCA(n_components=3)
data_pca = pca.fit_transform(data)

# 4. ЭТАП UMAP (с максимальным ускорением)
# n_neighbors=10 и n_epochs=200 для скорости
reducer = umap.UMAP(
    n_neighbors=10,
    min_dist=0.1,
    n_components=1,
    init='pca',
    n_epochs=200,
    low_memory=True
)
embedding = reducer.fit_transform(data_pca)

# 5. Сборка результатов обратно в картинки
out_h, out_w = h - patch_size[0] + 1, w - patch_size[1] + 1
pca_result = data_pca[:, 0].reshape(out_h, out_w) # 1-я компонента PCA
umap_result = embedding.reshape(out_h, out_w)     # Результат UMAP

# Визуализация
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

ax[0].imshow(image, cmap='gray')
ax[0].set_title('1. Оригинал')

ax[1].imshow(pca_result, cmap='gray')
ax[1].set_title('2. Структура через PCA (мгновенно)')

ax[2].imshow(umap_result, cmap='Spectral')
ax[2].set_title('3. Структура через UMAP (топология)')

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()