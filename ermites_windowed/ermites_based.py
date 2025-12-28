import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from numpy.fft import fft2, ifft2, fftfreq
import sys
import os

# Добавляем корневую папку проекта в sys.path для импорта модулей из base
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base.gradient_ops import poisson_solve, compute_gradients

# -----------------------------
# ПАРАМЕТРЫ
# -----------------------------
N = 100
sigma_blur = 5.0

W = 50          # окно
S = 10          # шаг
sigma1 = 1.5    # малый масштаб
alpha = 1.5     # усиление
eps = 0.3       # шаг in-place

# -----------------------------
# ЭРМИТОВЫ ФУНКЦИИ (1D)
# -----------------------------
def hermite_psi1(x, sigma):
    return (x / sigma) * np.exp(-x**2 / (2*sigma**2))

def hermite_psi3(x, sigma):
    # Используем полином "физиков" (2x^3 - 3x) для ортогональности с весом e^(-x^2)
    # (квадрат нашей огибающей e^(-x^2/2sigma^2) дает именно такой вес)
    return (2 * x**3 / sigma**3 - 3*x / sigma) * np.exp(-x**2 / (2*sigma**2))

def normalize(v):
    return v / np.sqrt(np.sum(v**2) + 1e-12)

def build_basis(W, sigma):
    x = np.arange(W) - W//2
    psi1 = normalize(hermite_psi1(x, sigma))
    psi3 = normalize(hermite_psi3(x, sigma))
    return psi1, psi3

psi1, psi3 = build_basis(W, sigma1)

# -----------------------------
# ИЗОБРАЖЕНИЯ A и B
# -----------------------------
A = np.zeros((N, N))
A[25, 25] = 1.0

B = gaussian_filter(A, sigma=sigma_blur)

# -----------------------------
# ГРАДИЕНТЫ
# -----------------------------
DBx, DBy = compute_gradients(B)

# -----------------------------
# ОКОННЫЙ АЛГОРИТМ (IN-PLACE)
# -----------------------------
def process_1d_inplace(G):
    G = G.copy()
    energy_log = []

    for i in range(0, G.shape[0]):
        for x0 in range(0, G.shape[1] - W, S):
            g = G[i, x0:x0+W]

            c1 = np.dot(g, psi1)
            c3 = np.dot(g, psi3)

            g_hat = alpha * (c1 * psi1 + c3 * psi3)

            # in-place обновление
            G[i, x0:x0+W] += eps * (g_hat - g)

            energy_log.append(np.sum(g_hat**2))

    print("Средняя энергия окна:", np.mean(energy_log))
    return G

print("Обработка DBx")
DBx_p = process_1d_inplace(DBx)

print("Обработка DBy")
DBy_p = process_1d_inplace(DBy.T).T



# -----------------------------
# ПУАССОН ЧЕРЕЗ GRADIENT_OPS
# -----------------------------
# Заменяем самописную функцию на проверенную из базы
C = poisson_solve(DBx_p, DBy_p)

# -----------------------------
# ВИЗУАЛИЗАЦИЯ
# -----------------------------
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.title("A (точка)")
plt.imshow(A, cmap='gray')

plt.subplot(2, 3, 2)
plt.title("B (размыто)")
plt.imshow(B, cmap='gray')

plt.subplot(2, 3, 3)
plt.title("DBx")
plt.imshow(DBx, cmap='gray')

plt.subplot(2, 3, 4)
plt.title("DBy")
plt.imshow(DBy, cmap='gray')

plt.subplot(2, 3, 5)
plt.title("C (восстановленное)")
plt.imshow(C, cmap='gray')

plt.tight_layout()
plt.show()
