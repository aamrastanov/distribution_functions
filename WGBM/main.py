import cv2
import numpy as np
import logging
import os
from processing_steps import (
    step_1_wavelet_decomposition,
    step_2_clean_ll_layer,
    step_3_4_wgm_reconstruction,
    step_5_inverse_wavelet
)

# Настройка подробного логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main(input_path, output_path):
    """
    Основной метод управления алгоритмом ВГМБ.
    input_path: путь к исходному PNG
    output_path: путь для сохранения результата
    """
    
    # --- КОНФИГУРАЦИЯ ПАРАМЕТРОВ ВГМБ ---
    K_VAL = 3          # k из примера
    STEP_VAL = 1.0     # step из примера
    NUM_BLOCKS = 6     # num_blocks из примера
    # ------------------------------------

    logging.info(f"--- СТАРТ ВГМБ: {input_path} ---")
    
    if not os.path.exists(input_path):
        logging.error(f"Файл {input_path} не найден в директории скрипта!")
        return

    # Загрузка изображения (в оттенках серого для работы с яркостным каналом)
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    img_float = img.astype(np.float32)

    try:
        # Шаг 1: Вейвлет-разложение (DWT)
        # Отделяем низкочастотный фон (муть) от деталей
        LL, details = step_1_wavelet_decomposition(img_float)

        # Шаг 2.1: Очистка слоя LL (удаление основной массы "тумана")
        LL_clean = step_2_clean_ll_layer(LL)
        
        # Шаг 2.2: Пороговая фильтрация деталей
        from processing_steps import step_2_threshold_details
        details_thresholded = step_2_threshold_details(details, threshold=10.0)

        # Шаг 3-4: Реконструкция деталей через строгий Гаус-базис
        # Передаем оригинальный LL для оценки карты прозрачности
        details_rec = step_3_4_wgm_reconstruction(
            details_thresholded,
            LL_original=LL,
            k=K_VAL, 
            step=STEP_VAL, 
            num_blocks=NUM_BLOCKS
        )

        # Шаг 5: Обратный синтез (IDWT)
        # Собираем очищенный фон и восстановленные детали вместе
        final_img = step_5_inverse_wavelet(LL_clean, details_rec)

        # Финальная нормализация и сохранение
        # Приводим значения обратно к диапазону 0-255 для записи в PNG
        result_u8 = cv2.normalize(final_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        cv2.imwrite(output_path, result_u8)
        logging.info(f"--- УСПЕХ: Результат сохранен в {output_path} ---")

    except Exception as e:
        logging.error(f"Критическая ошибка при выполнении ВГМБ: {e}", exc_info=True)

if __name__ == "__main__":
    # Определяем путь к файлу относительно скрипта
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(base_dir, 'input.png')
    output_file = os.path.join(base_dir, 'wgbm_restored_fixed.png')
    
    main(input_file, output_file)