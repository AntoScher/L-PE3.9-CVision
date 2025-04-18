import cv2
import numpy as np
import os

# ========== НАСТРОЙКИ ==========
window_name = "Ellipse Detection"
default_image = "IMG_cup.jpg"  # Убедитесь, что файл существует
output_dir = "results"  # Папка для сохранения результатов
# ===============================

# Создаем папку для результатов
os.makedirs(output_dir, exist_ok=True)

# Глобальные переменные
global_vars = {
    'thresh': None,
    'morph': None,
    'dilated': None,
    'result': None,
    'image': None,
    'gray': None
}

params = {
    'block_size': 55,
    'c': 9,
    'morph_size': 7,
    'min_area': 250,
    'aspect_ratio': 0.7,
    'angle_tolerance': 45,
    'dilate_iter': 2,
    'pre_blur': 5
}


def initialize_trackbars():
    """Инициализация трекбаров после создания окна"""
    cv2.createTrackbar('Block Size', window_name, 55, 100, lambda x: None)
    cv2.createTrackbar('C Constant', window_name, 9, 30, lambda x: None)
    cv2.createTrackbar('Morph Size', window_name, 7, 20, lambda x: None)
    cv2.createTrackbar('Min Area', window_name, 250, 1000, lambda x: None)
    cv2.createTrackbar('Aspect Ratio', window_name, 70, 100, lambda x: None)
    cv2.createTrackbar('Angle Tol.', window_name, 45, 90, lambda x: None)
    cv2.createTrackbar('Dilate Iter', window_name, 2, 5, lambda x: None)
    cv2.createTrackbar('Pre Blur', window_name, 5, 15, lambda x: None)


def update_parameters():
    """Безопасное обновление параметров"""
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
        # 1. Подготовка изображения
        blurred = cv2.GaussianBlur(global_vars['gray'],
                                   (params['pre_blur'], params['pre_blur']), 0)

        # 2. Адаптивная бинаризация
        thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            params['block_size'],
            params['c']
        )

        # 3. Морфологическая обработка
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (params['morph_size'], params['morph_size']))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        dilated = cv2.dilate(morph, kernel, iterations=params['dilate_iter'])

        # 4. Поиск контуров
        contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # 5. Поиск эллипса
        result = global_vars['image'].copy()
        best_ellipse = None

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < params['min_area'] or len(cnt) < 5:
                continue

            try:
                ellipse = cv2.fitEllipse(cnt)
                (_, _), (ma, MA), angle = ellipse
                aspect = min(ma, MA) / max(ma, MA)

                if (aspect >= params['aspect_ratio'] and
                        abs(angle) < params['angle_tolerance']):
                    best_ellipse = ellipse
                    cv2.ellipse(result, best_ellipse, (0, 255, 0), 2)
                    break

            except:
                continue

        # Сохраняем результаты
        global_vars.update({
            'thresh': thresh,
            'morph': morph,
            'dilated': dilated,
            'result': result
        })

        # 6. Сборка изображения для отображения
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
    image = cv2.resize(image, (640, 480))
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