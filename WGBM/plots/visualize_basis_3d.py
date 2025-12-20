
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import logging
# Импортируем низкоуровневую функцию генерации 1D сигнала, 
# чтобы мы могли сами задать разрешение
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from basis_engine import generate_f_n

def main():
    logging.basicConfig(level=logging.INFO)

    # Параметры из processing_steps.py
    k = 3
    step = 1.0
    num_blocks = 6
    
    # Фактор увеличения разрешения для гладкости графика
    resolution_factor = 5 
    
    # Эффективная сигма
    sigma = 1

    print(f"Generating High-Res WGM basis functions with: k={k}, step={step}, blocks={num_blocks}, sigma={sigma}")
    
    kernel_size = int(step * k * (num_blocks + 1))
    print(f"Kernel size: {kernel_size}")
    print(f"Visualization Resolution Factor: {resolution_factor}x")
    
    # Создаем густую сетку координат для гладкости
    # Вместо 25 точек берем 250
    num_points = kernel_size * resolution_factor
    x = np.linspace(0, kernel_size, num_points)
    y = np.linspace(0, kernel_size, num_points)
    X, Y = np.meshgrid(x, y)

    # Создаем фигуру с 3D подграфиками
    # k=3 -> 4 сдвига (0, 1, 2, 3)
    num_kernels = k + 1 
    fig = plt.figure(figsize=(5 * num_kernels, 5))
    
    for s in range(num_kernels):
        # Генерируем 1D функцию на густой сетке
        f_n = generate_f_n(x, shift_index=s, k=k, step=step, sigma=sigma, num_blocks=num_blocks)
        
        # Создаем 2D ядро (внешнее произведение)
        kernel_high_res = np.outer(f_n, f_n)
        
        # Нормировка (для визуализации) аналогично оригинальной функции
        if np.max(kernel_high_res) > 0:
            kernel_high_res /= np.max(kernel_high_res)

        # Добавляем 3D subplot
        ax = fig.add_subplot(1, num_kernels, s+1, projection='3d')
        
        # Строим glадкую поверхность
        # rcount/ccount или stride управляют даунсэмплингом для отрисовки,
        # чтобы matplotlib не тормозил на миллионах полигонов, но при этом форма была гладкой.
        surf = ax.plot_surface(X, Y, kernel_high_res, cmap=cm.viridis,
                               linewidth=0, antialiased=False,
                               rcount=100, ccount=100) # Отрисовываем сеткой 100x100 для скорости
        
        ax.set_title(f'Kernel #{s}\nShift index: {s}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Amplitude')
        
        ax.view_init(elev=30, azim=45)

    plt.suptitle(f'Smooth 3D Visualization (High-Res)\nk={k}, blocks={num_blocks}, step={step}')
    
    print("Displaying smooth 3D plot...")
    plt.show()

if __name__ == "__main__":
    main()
