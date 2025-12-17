import cv2
import pywt
import numpy as np
import logging
from basis_engine import generate_wgm_kernel

def step_1_wavelet_decomposition(image):
    logging.info("Шаг 1: Вейвлет-разложение (DWT)")
    coeffs = pywt.dwt2(image, 'db1')
    LL, (LH, HL, HH) = coeffs
    return LL, (LH, HL, HH)

def step_2_clean_ll_layer(LL):
    logging.info("Шаг 2: Очистка низкочастотного слоя (LL)")
    # Применяем адаптивное выравнивание гистограммы для проявления дна
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    LL_uint8 = cv2.normalize(LL, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    LL_enhanced = clahe.apply(LL_uint8)
    return LL_enhanced.astype(np.float32)

def step_3_gaussian_reconstruction(details, kernel_size=15):
    logging.info("Шаг 3-4: Проекция на Гаус-базис и усиление текстур")
    LH, HL, HH = details
    # Генерируем наше специфическое ядро из документа
    kernel = generate_wgm_kernel(kernel_size, k_blocks=3, sigma=1.5)
    
    # Восстанавливаем детали через свертку с ВГМБ-ядром
    # Это имитирует "подбор амплитуды сгустков"
    LH_rec = cv2.filter2D(LH, -1, kernel) * 1.5
    HL_rec = cv2.filter2D(HL, -1, kernel) * 1.5
    HH_rec = cv2.filter2D(HH, -1, kernel) * 1.5
    
    return LH_rec, HL_rec, HH_rec

def step_5_inverse_wavelet(LL, details):
    logging.info("Шаг 5: Обратное вейвлет-преобразование (IDWT)")
    return pywt.idwt2((LL, details), 'db1')