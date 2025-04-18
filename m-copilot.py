import cv2
import numpy as np
import os

# ========= НАСТРОЙКИ ============
window_name = "Ellipse Detection"
default_image = "IMG_cup.jpg"  # Убедитесь, что файл существует
output_dir = "results"  # Папка для сохранения результатов
# ================================

os.makedirs(output_dir, exist_ok=True)

global_vars = {
    'image': None,
    'gray': None,
    'thresh': None,
    'morph': None,
    'dilated': None,
    'result': None
}

params = {
    'pre_blur': 5,  # Размер ядра для Гауссова размытия (нечётное число)
    'morph_size': 7,  # Размер ядра для морфологических операций
    'min_area': 250,  # Минимальная площадь контура для фильтрации
    'aspect_ratio': 0.7,  # Минимальное допустимое соотношение сторон эллипса
    'angle_tolerance': 45,  # Допустимый угол наклона эллипса
    'dilate_iter': 2  # Число итераций для дилатации
}


def initialize_trackbars():
    # Трекбары для динамической настройки параметров обработки
    cv2.createTrackbar('Pre Blur', window_name, params['pre_blur'], 15, lambda x: None)
    cv2.createTrackbar('Morph Size', window_name, params['morph_size'], 20, lambda x: None)
    cv2.createTrackbar('Min Area', window_name, params['min_area'], 1000, lambda x: None)
    cv2.createTrackbar('Aspect Ratio', window_name, int(params['aspect_ratio'] * 100), 100, lambda x: None)
    cv2.createTrackbar('Angle Tol.', window_name, params['angle_tolerance'], 90, lambda x: None)
    cv2.createTrackbar('Dilate Iter', window_name, params['dilate_iter'], 5, lambda x: None)


def update_parameters():
    try:
        # Обновление параметров из позиций трекбаров
        params['pre_blur'] = cv2.getTrackbarPos('Pre Blur', window_name) | 1  # Гарантируем, что число нечётное
        params['morph_size'] = max(1, cv2.getTrackbarPos('Morph Size', window_name))
        params['min_area'] = cv2.getTrackbarPos('Min Area', window_name)
        params['aspect_ratio'] = cv2.getTrackbarPos('Aspect Ratio', window_name) / 100
        params['angle_tolerance'] = cv2.getTrackbarPos('Angle Tol.', window_name)
        params['dilate_iter'] = cv2.getTrackbarPos('Dilate Iter', window_name)
    except cv2.error:
        pass


def process_image():
    try:
        # Повышение контраста с помощью CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(global_vars['gray'])
        # Гауссово размытие для снижения шума
        blurred = cv2.GaussianBlur(enhanced_gray, (params['pre_blur'], params['pre_blur']), 0)

        # Пороговая обработка с использованием метода Otsu
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Морфологическая обработка: замыкание для устранения разрывов
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params['morph_size'], params['morph_size']))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        dilated = cv2.dilate(morph, kernel, iterations=params['dilate_iter'])

        # Поиск контуров
        contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        result = global_vars['image'].copy()
        best_ellipse = None

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < params['min_area'] or len(cnt) < 5:
                continue

            # Пропускаем выпуклые контуры, которые обычно не дают корректного эллипса
            if cv2.isContourConvex(cnt):
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            compactness = (4 * np.pi * area) / (perimeter ** 2)
            if compactness < 0.7:  # Фильтрация по компактности (настройте порог по необходимости)
                continue

            try:
                ellipse = cv2.fitEllipse(cnt)
                (_, _), (ma, MA), angle = ellipse
                aspect = min(ma, MA) / max(ma, MA)
                if aspect >= params['aspect_ratio'] and abs(angle) < params['angle_tolerance']:
                    best_ellipse = ellipse
                    cv2.ellipse(result, best_ellipse, (0, 255, 0), 2)
                    break
            except cv2.error:
                continue

        global_vars.update({
            'thresh': thresh,
            'morph': morph,
            'dilated': dilated,
            'result': result
        })

        # Собираем изображение для отображения: оригинал, пороговое, после морфологии и результат наложения эллипса
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
    try:
        # Проверка и сохранение всех необходимых данных
        required_keys = ['image', 'thresh', 'morph', 'dilated', 'result']
        for key in required_keys:
            if global_vars[key] is None:
                raise ValueError(f"Данные {key} не инициализированы")
            if not isinstance(global_vars[key], np.ndarray):
                raise ValueError(f"Данные {key} не являются изображением")

        cv2.imwrite(os.path.join(output_dir, "1_original.jpg"), global_vars['image'])
        cv2.imwrite(os.path.join(output_dir, "2_threshold.jpg"), global_vars['thresh'])
        cv2.imwrite(os.path.join(output_dir, "3_morphology.jpg"), global_vars['morph'])
        cv2.imwrite(os.path.join(output_dir, "4_dilated.jpg"), global_vars['dilated'])
        cv2.imwrite(os.path.join(output_dir, "5_result.jpg"), global_vars['result'])
        print(f"Файлы успешно сохранены в: {os.path.abspath(output_dir)}")
    except Exception as e:
        print(f"Ошибка сохранения: {str(e)}")


def main():
    if not os.path.exists(default_image):
        print(f"Файл {default_image} не найден!")
        return

    image = cv2.imread(default_image)
    if image is None:
        print("Ошибка чтения файла изображения!")
        return

    image = cv2.resize(image, (640, 480))
    global_vars['image'] = image
    global_vars['gray'] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    initialize_trackbars()

    while True:
        try:
            update_parameters()
            process_image()
            key = cv2.waitKey(1) & 0xFF

            # Сохранение результатов по нажатию клавиши 's'
            if key == ord('s'):
                save_results()
            # Выход при нажатии ESC
            elif key == 27:
                break
        except KeyboardInterrupt:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()