import cv2
import pywt
import numpy as np
import logging
from basis_engine import get_wgm_basis_matrix

def step_1_wavelet_decomposition(image):
    logging.info("Шаг 1: DWT разложение")
    coeffs = pywt.dwt2(image, 'db1')
    return coeffs # LL, (LH, HL, HH)

def step_2_clean_ll_layer(LL):
    logging.info("Шаг 2: Адаптивная очистка мути в LL слое")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    LL_u8 = cv2.normalize(LL, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return clahe.apply(LL_u8).astype(np.float32)

def step_3_4_wgm_reconstruction(details, k, step, num_blocks):
    logging.info(f"Шаг 3-4: Проекция на семейство функций F_n (k={k}, blocks={num_blocks})")
    LH, HL, HH = details
    
    # Генерируем набор ядер (F1, F2, F3...)
    kernels = get_wgm_basis_matrix(kernel_size=31, k=k, step=step, num_blocks=num_blocks)
    
    # Результирующие слои (аккумуляторы восстановления)
    LH_final = np.zeros_like(LH)
    HL_final = np.zeros_like(HL)
    HH_final = np.zeros_like(HH)
    
    # Проходим каждым ядерным сдвигом по деталям
    # Это реализует идею разложения по базису: находим максимум отклика
    for kernel in kernels:
        LH_final += cv2.filter2D(LH, -1, kernel)
        HL_final += cv2.filter2D(HL, -1, kernel)
        HH_final += cv2.filter2D(HH, -1, kernel)
        
    # Усредняем и применяем Gain (усиление)
    gain = 2.0
    return (LH_final * gain / len(kernels), 
            HL_final * gain / len(kernels), 
            HH_final * gain / len(kernels))

def step_5_inverse_wavelet(LL, details):
    logging.info("Шаг 5: Сборка IDWT")
    return pywt.idwt2((LL, details), 'db1')