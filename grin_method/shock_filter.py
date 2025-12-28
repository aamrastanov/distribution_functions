
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, laplace
from mpl_toolkits.mplot3d import Axes3D

# --- ПАРАМЕТРЫ ---
N = 100
sigma_blur_real = 5.0
num_points = 15
num_iter = 30        # Количество итераций (шагов времени)
dt = 0.5              # Шаг времени (устойчивость обычно при dt < 1.0)

# --- 1. ГЕНЕРАЦИЯ ДАННЫХ ---
np.random.seed(42)
A = np.zeros((N, N))
for _ in range(num_points):
    ry = np.random.randint(10, N-10)
    rx = np.random.randint(10, N-10)
    A[ry, rx] = 1.0

# Размытие
B = gaussian_filter(A, sigma=sigma_blur_real, mode='constant', cval=0.0)

# --- 2. УЛУЧШЕННЫЙ ШОК-ФИЛЬТР ---
Restored = B.copy()
sigma_guidance = 2.0  # Сглаживание для поиска "настоящих" склонов
eps = 1e-4             # Порог чувствительности

print(f"Запуск стабилизированного Шок-фильтра: {num_iter} итераций...")

for i in range(num_iter):
    # 1. Лапласиан считаем на СГЛАЖЕННОЙ версии (Guidance)
    # Это ключевой момент, чтобы не плодить ложные пики из шума
    guidance = gaussian_filter(Restored, sigma=sigma_guidance)
    L = laplace(guidance)
    
    # 2. Магнитуду градиента считаем через центральные разности (или Upwind)
    gy, gx = np.gradient(Restored)
    grad_mag = np.sqrt(gx**2 + gy**2)
    
    # 3. Обновление только там, где сигнал значимый
    # Мы используем sign(L) только если Лапласиан достаточно велик
    update_mask = np.abs(L) > eps
    
    # Рекурсивный шаг
    Restored[update_mask] -= dt * np.sign(L[update_mask]) * grad_mag[update_mask]
    
    # Ограничение физичности
    Restored = np.maximum(Restored, 0)
    
    if (i+1) % 10 == 0:
        print(f"Итерация {i+1}/{num_iter} завершена.")

# --- 3. ВИЗУАЛИЗАЦИЯ ---
fig = plt.figure(figsize=(24, 8))
X_grid, Y_grid = np.meshgrid(np.arange(N), np.arange(N))

# 1. Original (3D)
ax1 = fig.add_subplot(131, projection='3d')
surf1 = ax1.plot_surface(X_grid, Y_grid, A, cmap='gray', linewidth=0, antialiased=False)
ax1.set_title("Original (Point Sources)")

# 2. Blurred (3D)
ax2 = fig.add_subplot(132, projection='3d')
surf2 = ax2.plot_surface(X_grid, Y_grid, B, cmap='gray', linewidth=0, antialiased=False)
ax2.set_title(f"Blurred (Sigma={sigma_blur_real})")

# 3. Restored (3D - Shock Filter)
ax3 = fig.add_subplot(133, projection='3d')
surf3 = ax3.plot_surface(X_grid, Y_grid, Restored, cmap='hot', linewidth=0, antialiased=False)
ax3.set_title(f"Shock Filter ({num_iter} iter, dt={dt})")

plt.tight_layout()
plt.show()

# 2D Comparison for contrast
plt.figure(figsize=(15, 5))
plt.subplot(131); plt.imshow(A, cmap='gray'); plt.title('Original')
plt.subplot(132); plt.imshow(B, cmap='gray'); plt.title('Blurred')
plt.subplot(133); plt.imshow(Restored, cmap='hot'); plt.title('Shock Filter')
plt.tight_layout()
plt.show()
