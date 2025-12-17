import cv2
import numpy as np
import logging
import os
from processing_steps import (
    step_1_wavelet_decomposition,
    step_2_clean_ll_layer,
    step_3_gaussian_reconstruction,
    step_5_inverse_wavelet
)

# Настройка подробного логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main(input_path, output_path):
    logging.info(f"Запуск процесса восстановления для файла: {input_path}")
    
    if not os.path.exists(input_path):
        logging.error("Файл не найден!")
        return

    # Загрузка изображения
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    img_float = img.astype(np.float32)

    try:
        # 1. Разложение
        LL, (LH, HL, HH) = step_1_wavelet_decomposition(img_float)

        # 2. Очистка мути в LL
        LL_clean = step_2_clean_ll_layer(LL)

        # 3. Реконструкция деталей через Гаус-базис
        LH_r, HL_r, HH_r = step_3_gaussian_reconstruction((LH, HL, HH))

        # 5. Сборка
        result = step_5_inverse_wavelet(LL_clean, (LH_r, HL_r, HH_r))

        # Финальная нормализация для сохранения в PNG
        result_final = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        cv2.imwrite(output_path, result_final)
        logging.info(f"Восстановление успешно завершено. Результат: {output_path}")

    except Exception as e:
        logging.error(f"Ошибка в ходе выполнения алгоритма: {e}")

if __name__ == "__main__":
    # Определяем путь к файлу относительно скрипта
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(base_dir, 'input.png')
    output_file = os.path.join(base_dir, 'wgbm_restored.png')
    
    main(input_file, output_file)