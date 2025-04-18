import cv2
import numpy as np
import os

# ========== НАСТРОЙКИ ==========
window_name = "Ellipse Detection"
default_image = "IMG_cup.jpg"  # Убедитесь, что файл существует
output_dir = "results"  # Папка для сохранения результатов
# ===============================

os.makedirs(output_dir, exist_ok=True)

global_vars = {
    'thresh': None,
    'morph': None,
    'dilated': None,
    'result': None,
    'image': None,
    'gray': None
}

params = {
    'block_size': 21,  # Начальное значение увеличено
    'c': 15,  # Увеличено для более темного фона
    'morph_size': 5,  # Уменьшено для точности
    'min_area': 500,  # Повышена минимальная площадь
    'aspect_ratio': 0.75,
    'angle_tolerance': 45,
    'dilate_iter': 2,
    'pre_blur': 9  # Увеличено для лучшего сглаживания
}


def initialize_trackbars():
    cv2.createTrackbar('Block Size', window_name, 21, 100, lambda x: None)
    cv2.createTrackbar('C Constant', window_name, 15, 50, lambda x: None)
    cv2.createTrackbar('Morph Size', window_name, 5, 20, lambda x: None)
    cv2.createTrackbar('Min Area', window_name, 500, 5000, lambda x: None)
    cv2.createTrackbar('Aspect Ratio', window_name, 75, 100, lambda x: None)
    cv2.createTrackbar('Angle Tol.', window_name, 45, 90, lambda x: None)
    cv2.createTrackbar('Dilate Iter', window_name, 2, 10, lambda x: None)
    cv2.createTrackbar('Pre Blur', window_name, 9, 15, lambda x: None)


def update_parameters():
    try:
        params.update({
            'block_size': max(3, cv2.getTrackbarPos('Block Size', window_name) | 1),
            'c': cv2.getTrackbarPos('C Constant', window_name),
            'morph_size': max(1, cv2.getTrackbarPos('Morph Size', window_name)),
            'min_area': cv2.getTrackbarPos('Min Area', window_name),
            'aspect_ratio': cv2.getTrackbarPos('Aspect Ratio', window_name) / 100,
            'angle_tolerance': cv2.getTrackbarPos('Angle Tol.', window_name),
            'dilate_iter': cv2.getTrackbarPos('Dilate Iter', window_name),
            'pre_blur': cv2.getTrackbarPos('Pre Blur', window_name) | 1
        })
    except cv2.error:
        pass


def process_image():
    try:
        # 1. Улучшенное сглаживание
        blurred = cv2.bilateralFilter(global_vars['gray'],
                                      params['pre_blur'],
                                      75, 75)  # Сохраняет края

        # 2. Улучшенная бинаризация
        thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            params['block_size'],
            params['c']
        )

        # 3. Улучшенная морфология
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (params['morph_size'], params['morph_size']))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)  # Сначала удаляем шум
        dilated = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel,
                                   iterations=params['dilate_iter'])  # Затем закрываем

        # 4. Поиск контуров
        contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        result = global_vars['image'].copy()
        best_ellipse = None
        best_area = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)

            # Пропускаем маленькие/неподходящие контуры
            if perimeter == 0 or area < params['min_area'] or len(cnt) < 5:
                continue

            circularity = (4 * np.pi * area) / (perimeter ** 2)

            # Новые условия отбора
            if circularity < 0.5 or area < params['min_area']:
                continue

            try:
                ellipse = cv2.fitEllipse(cnt)
                (x, y), (ma, MA), angle = ellipse
                aspect = min(ma, MA) / max(ma, MA)
                area_ratio = area / (np.pi * ma * MA / 4)  # Соотношение реальной и идеальной площади

                # Комплексные условия отбора
                if (aspect >= params['aspect_ratio'] and
                        area_ratio > 0.6 and
                        abs(angle) < params['angle_tolerance']):

                    # Выбираем самый большой подходящий эллипс
                    if area > best_area:
                        best_ellipse = ellipse
                        best_area = area

            except:
                continue

        # Рисуем лучший эллипс
        if best_ellipse is not None:
            cv2.ellipse(result, best_ellipse, (0, 255, 0), 2)

        # Обновляем глобальные переменные
        global_vars.update({
            'thresh': thresh,
            'morph': morph,
            'dilated': dilated,
            'result': result
        })

        # Собираем изображение для отображения
        display_images = [
            global_vars['image'],
            cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR),
            result
        ]

        top = np.hstack(display_images[:2])
        bottom = np.hstack(display_images[2:])
        combined = np.vstack([top, bottom])

        cv2.imshow(window_name, combined)

    except Exception as e:
        print(f"Ошибка обработки: {str(e)}")

# Остальные функции (save_results, main) остаются без изменений
# ... (вставьте их из предыдущей версии кода)
def save_results():
    """Безопасное сохранение результатов"""
    try:
        # Явная проверка на инициализацию всех необходимых данных
        required_keys = ['image', 'thresh', 'morph', 'dilated', 'result']
        for key in required_keys:
            if global_vars[key] is None:
                raise ValueError(f"Данные {key} не инициализированы")

        # Проверка, что это действительные изображения
        for key in required_keys:
            if not isinstance(global_vars[key], np.ndarray):
                raise ValueError(f"Данные {key} не являются изображением")

        # Сохранение
        cv2.imwrite(os.path.join(output_dir, "1_original.jpg"), global_vars['image'])
        cv2.imwrite(os.path.join(output_dir, "2_threshold.jpg"), global_vars['thresh'])
        cv2.imwrite(os.path.join(output_dir, "3_morphology.jpg"), global_vars['morph'])
        cv2.imwrite(os.path.join(output_dir, "4_dilated.jpg"), global_vars['dilated'])
        cv2.imwrite(os.path.join(output_dir, "5_result.jpg"), global_vars['result'])

        print(f"Файлы успешно сохранены в: {os.path.abspath(output_dir)}")

    except Exception as e:
        print(f"Ошибка сохранения: {str(e)}")

def main():
    # Проверка существования файла
    if not os.path.exists(default_image):
        print(f"Файл {default_image} не найден!")
        return

    # Загрузка изображения
    image = cv2.imread(default_image)
    if image is None:
        print("Ошибка чтения файла изображения!")
        return

    # Предварительная обработка
    image = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
    global_vars['image'] = image
    global_vars['gray'] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Создание интерфейса
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    initialize_trackbars()

    # Основной цикл
    while True:
        try:
            update_parameters()
            process_image()

            key = cv2.waitKey(1) & 0xFF

            # Обработка клавиши 'S' (английская раскладка)
            if key == ord('s'):
                save_results()

            # Выход по ESC
            elif key == 27:
                break

        except KeyboardInterrupt:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
