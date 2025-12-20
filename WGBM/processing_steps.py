import cv2
import pywt
import numpy as np
import logging
from basis_engine import get_wgm_basis_matrix

def step_1_wavelet_decomposition(image):
    logging.info("Шаг 1: DWT разложение")
    coeffs = pywt.dwt2(image, 'db1')
    return coeffs # LL, (LH, HL, HH)

def step_2_threshold_details(details, threshold=10.0):
    """
    Шаг 2.2: Пороговая фильтрация деталей для удаления шума.
    Обнуляет коэффициенты меньше порога.
    """
    logging.info(f"Шаг 2.2: Пороговая фильтрация (Hard Thresholding, T={threshold})")
    LH, HL, HH = details
    
    # Hard thresholding
    LH_clean = pywt.threshold(LH, threshold, mode='hard')
    HL_clean = pywt.threshold(HL, threshold, mode='hard')
    HH_clean = pywt.threshold(HH, threshold, mode='hard')
    
    return (LH_clean, HL_clean, HH_clean)

def step_2_clean_ll_layer_clahe(LL):
    logging.info("Шаг 2.1: Адаптивная очистка мути в LL слое (CLAHE) [Unused]")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    LL_u8 = cv2.normalize(LL, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # Возвращаем float32 для дальнейшей работы
    return clahe.apply(LL_u8).astype(np.float32)

def step_2_clean_ll_layer_dcp(LL):
    """
    Шаг 2.1: Очистка LL слоя с использованием упрощенного Dark Channel Prior (DCP).
    Модель: I(x) = J(x)t(x) + A(1-t(x))
    Цель: восстановить J(x).
    """
    logging.info("Шаг 2.1: Адаптивная очистка мути в LL слое (Simplified DCP)")
    
    # 1. Оценка атмосферного света (A) и Dark Channel
    # Для одного канала Dark Channel - это минимум в окрестности
    patch_size = 15
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark_channel = cv2.erode(LL, kernel)
    
    # A - это интенсивность самых ярких пикселей в темном канале (или просто макс)
    # Берем макс из темного канала как грубую оценку A
    A = np.max(dark_channel)
    if A == 0: A = 1.0 # Защита
    
    # 2. Оценка Transmission Map t(x)
    # t(x) = 1 - omega * min(I/A)
    # min(I) это и есть dark_channel
    omega = 0.9
    t = 1.0 - omega * (dark_channel / A)
    
    # Ограничение t снизу, чтобы не делить на 0 и не усиливать шум слишком сильно
    t0 = 0.1
    t = np.maximum(t, t0)
    
    # 3. Восстановление сцены (Radiance Recover)
    # J = (I - A)/t + A
    J = (LL - A) / t + A
    
    return J

def estimate_transmission(LL):
    """
    Оценка карты пропускания T(x) на основе LL слоя.
    Предполагаем, что более яркие участки LL соответствуют более густой мути/рассеянию
    (если это подводное изображение с искусственным светом) или наоборот.
    
    В классической модели: J = I/t - A/t + A.
    Здесь используем простую эвристику: T(x) ~ 1 - normalized(LL).
    Чем ярче фоновая муть (LL), тем меньше прозрачность.
    """
    LL_norm = cv2.normalize(LL, None, 0.0, 1.0, cv2.NORM_MINMAX)
    # T(x) должна быть в диапазоне (0, 1].
    # Инвертируем: ярко -> мутно -> малое T.
    T = 1.0 - LL_norm * 0.9 # *0.9 чтобы T не уходило в полный 0
    return T

def step_3_4_wgm_reconstruction(details, LL_original):
    """
    Шаг 3-4: Полный цикл Анализ - Усиление - Синтез.
    
    1. Анализ: Проецируем детали на базис -> получаем коэффициенты C_k.
    2. Усиление: C_k' = C_k * Gain(x), где Gain(x) ~ 1/T(x).
    3. Синтез: Восстанавливаем детали, проецируя C_k' обратно через базис.
    """
    LH, HL, HH = details
    
    # Генерируем набор ядер (F1, F2, F3...)
    # kernel_size должен быть достаточным, чтобы вместить функции
    # k=3, num_blocks=6, step=1 -> эффективная ширина ~ 22. 25 с небольшим запасом.

    # --- КОНФИГУРАЦИЯ ПАРАМЕТРОВ ВГМБ ---
    k = 15          # k из примера
    step = 1.0     # step из примера
    num_blocks = 1     # num_blocks из примера
    sigma = 1    
    # ------------------------------------

    logging.info(f"Шаг 3-4: Реконструкция WGM (k={k}, blocks={num_blocks})")

    kernels = get_wgm_basis_matrix(sigma=sigma, k=k, step=step, num_blocks=num_blocks)
    
    # Карта прозрачности и усиления
    T = estimate_transmission(LL_original)
    # Избегаем деления на 0
    epsilon = 0.1
    Gain_map = 1.0 / (T + epsilon)
    
    # Поскольку размер LL совпадает с размером деталей в DWT (для db1, если размеры кратны 2),
    # используем Gain_map прямо так. Или ресайзим если нужно.
    # DWT делит размер картинки на 2. LL и детали одного размера.
    if Gain_map.shape != LH.shape:
        Gain_map = cv2.resize(Gain_map, (LH.shape[1], LH.shape[0]))

    # Аккумуляторы для восстановленных деталей
    LH_rec = np.zeros_like(LH)
    HL_rec = np.zeros_like(HL)
    HH_rec = np.zeros_like(HH)
    
    norm_factor = 0.001 # Коэффициент для подавления энергии после двойной свертки
    
    # --- ЦИКЛ ПО БАЗИСНЫМ ФУНКЦИЯМ ---
    for i, kernel in enumerate(kernels):
        # 1. АНАЛИЗ (Analysis): Получение коэффициентов C_k
        # Используем корреляцию (или свертку с перевернутым ядром, но ядро симметрично относительно формы, хотя F(x) не симметрична)
        # В pdf: "Projections". Это скалярное произведение. Реализуется как фильтрация.
        
        C_LH = cv2.filter2D(LH, -1, kernel)
        C_HL = cv2.filter2D(HL, -1, kernel)
        C_HH = cv2.filter2D(HH, -1, kernel)
        
        # 2. УСИЛЕНИЕ (Gain Adjustment)
        # Применяем карту усиления локально
        C_LH_mod = C_LH * Gain_map
        C_HL_mod = C_HL * Gain_map
        C_HH_mod = C_HH * Gain_map
        
        # 3. СИНТЕЗ (Synthesis): Восстановление
        # Проецируем модифицированные коэффициенты обратно через те же базисные функции
        # Rec += C_k * Basis_k
        
        LH_rec += cv2.filter2D(C_LH_mod, -1, kernel)
        HL_rec += cv2.filter2D(C_HL_mod, -1, kernel)
        HH_rec += cv2.filter2D(C_HH_mod, -1, kernel)

    # Приводим энергию в порядок (эмпирическая нормировка, т.к. базис не ортонормирован строго)
    # Делим на количество ядер и еще на коэффициент
    LH_rec *= norm_factor
    HL_rec *= norm_factor
    HH_rec *= norm_factor

    return (LH_rec, HL_rec, HH_rec)

def step_5_inverse_wavelet(LL, details):
    logging.info("Шаг 5: Сборка IDWT")
    return pywt.idwt2((LL, details), 'db1')

def step_5_post_processing(image):
    """
    Финальная постобработка:
    1. Гамма-коррекция для осветления средних тонов (вода часто темная).
    2. Выравнивание гистограммы.
    """
    # logging.info("Шаг 5.3: Финальная коррекция") [DISABLED]
    return image