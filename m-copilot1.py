import cv2
import numpy as np
import os

# ========= НАСТРОЙКИ ============
window_name = "Ellipse Detection with Canny"
default_image = "IMG_cup.jpg"  # Убедитесь, что файл существует
output_dir = "results"         # Папка для сохранения результатов

# Создаем папку для результатов, если она не существует
os.makedirs(output_dir, exist_ok=True)

# Глобальные переменные для хранения изображений на разных этапах
global_vars = {
    'image': None,
    'gray': None,
    'edges': None,
    'morph': None,
    'result': None
}

# Параметры обработки с базовыми значениями
params = {
    'pre_blur': 5,          # Размер ядра для Гауссова размытия (должно быть нечётным)
    'morph_size': 5,        # Размер ядра для морфологических операций
    'min_area': 250,        # Минимальная площадь контура для фильтрации
    'aspect_ratio': 0.7,    # Минимальное допустимое соотношение сторон эллипса
    'angle_tolerance': 45,  # Допустимый угол наклона эллипса (в градусах)
    'dilate_iter': 1,       # Число итераций дилатации для границ
    'canny_low': 50,        # Нижний порог для Canny
    'canny_high': 150       # Верхний порог для Canny
}

def initialize_trackbars():
    # Трекбары для динамической настройки параметров
    cv2.createTrackbar('Pre Blur', window_name, params['pre_blur'], 15, lambda x: None)
    cv2.createTrackbar('Morph Size', window_name, params['morph_size'], 20, lambda x: None)
    cv2.createTrackbar('Min Area', window_name, params['min_area'], 1000, lambda x: None)
    cv2.createTrackbar('Aspect Ratio', window_name, int(params['aspect_ratio'] * 100), 100, lambda x: None)
    cv2.createTrackbar('Angle Tol.', window_name, params['angle_tolerance'], 90, lambda x: None)
    cv2.createTrackbar('Dilate Iter', window_name, params['dilate_iter'], 5, lambda x: None)
    cv2.createTrackbar('Canny Low', window_name, params['canny_low'], 300, lambda x: None)
    cv2.createTrackbar('Canny High', window_name, params['canny_high'], 300, lambda x: None)

def update_parameters():
    try:
        # Обновление параметров из позиций трекбаров
        params['pre_blur'] = cv2.getTrackbarPos('Pre Blur', window_name) | 1  # Гарантируем, что число нечётное
        params['morph_size'] = max(1, cv2.getTrackbarPos('Morph Size', window_name))
        params['min_area'] = cv2.getTrackbarPos('Min Area', window_name)
        params['aspect_ratio'] = cv2.getTrackbarPos('Aspect Ratio', window_name) / 100.0
        params['angle_tolerance'] = cv2.getTrackbarPos('Angle Tol.', window_name)
        params['dilate_iter'] = cv2.getTrackbarPos('Dilate Iter', window_name)
        params['canny_low'] = cv2.getTrackbarPos('Canny Low', window_name)
        params['canny_high'] = cv2.getTrackbarPos('Canny High', window_name)
    except cv2.error:
        pass

def process_image():
    try:
        # 1. Повышаем контрастность с помощью CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(global_vars['gray'])

        # 2. Применяем Гауссово размытие для снижения шума
        blurred = cv2.GaussianBlur(enhanced_gray, (params['pre_blur'], params['pre_blur']), 0)

        # 3. Детектор границ Canny с порогами, заданными через трекбары
        edges = cv2.Canny(blurred, params['canny_low'], params['canny_high'])

        # 4. Усиливаем края с помощью дилатации
        kernel_edges = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges_dilated = cv2.dilate(edges, kernel_edges, iterations=params['dilate_iter'])

        # 5. Применяем морфологическую операцию (закрытие), чтобы устранить разрывы в границах
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params['morph_size'], params['morph_size']))
        morph = cv2.morphologyEx(edges_dilated, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 6. Поиск контуров на основе обработанного изображения
        contours, _ = cv2.findContours(morph, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        result = global_vars['image'].copy()
        best_ellipse = None

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < params['min_area'] or len(cnt) < 5:
                continue

            # Пропускаем выпуклые контуры, которые могут не давать корректный эллипс
            if cv2.isContourConvex(cnt):
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            # Вычисляем компактность контура
            compactness = (4 * np.pi * area) / (perimeter ** 2)
            if compactness < 0.7:
                continue

            try:
                ellipse = cv2.fitEllipse(cnt)
                (_, _), (ma, MA), angle = ellipse
                aspect = min(ma, MA) / max(ma, MA)
                if aspect >= params['aspect_ratio'] and abs(angle) < params['angle_tolerance']:
                    best_ellipse = ellipse
                    cv2.ellipse(result, ellipse, (0, 255, 0), 2)
                    break
            except cv2.error:
                continue

        # Обновляем глобальные переменные для отображения результатов
        global_vars.update({
            'edges': edges,
            'morph': morph,
            'result': result
        })

        # 7. Собираем изображения для отображения: оригинал, Canny, морфология и итоговый результат
        display_images = [
            global_vars['image'],
            cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR),
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
    """Сохраняем все промежуточные результаты в указанной папке."""
    try:
        required_keys = ['image', 'edges', 'morph', 'result']
        for key in required_keys:
            if global_vars.get(key) is None or not isinstance(global_vars[key], np.ndarray):
                raise ValueError(f"Данные {key} не инициализированы корректно")

        cv2.imwrite(os.path.join(output_dir, "1_original.jpg"), global_vars['image'])
        cv2.imwrite(os.path.join(output_dir, "2_edges.jpg"), global_vars['edges'])
        cv2.imwrite(os.path.join(output_dir, "3_morphology.jpg"), global_vars['morph'])
        cv2.imwrite(os.path.join(output_dir, "4_result.jpg"), global_vars['result'])
        print(f"Файлы успешно сохранены в: {os.path.abspath(output_dir)}")
    except Exception as e:
        print(f"Ошибка сохранения: {str(e)}")

def main():
    # Проверка существования файла изображения
    if not os.path.exists(default_image):
        print(f"Файл {default_image} не найден!")
        return

    # Загрузка и предварительная обработка изображения
    image = cv2.imread(default_image)
    if image is None:
        print("Ошибка чтения изображения!")
        return

    # Изменяем размер для удобства обработки
    image = cv2.resize(image, (640, 480))
    global_vars['image'] = image
    global_vars['gray'] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Создаем окно для отображения и настраиваем трекбары
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    initialize_trackbars()

    while True:
        update_parameters()
        process_image()

        key = cv2.waitKey(1) & 0xFF

        # Сохранение результатов по нажатию 's'
        if key == ord('s'):
            save_results()
        # Выход при нажатии клавиши ESC
        elif key == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()