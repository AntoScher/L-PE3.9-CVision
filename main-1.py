import cv2
import matplotlib
from matplotlib import pyplot as plt
import numpy as np

# Настройка отображения графиков
matplotlib.rcParams['figure.figsize'] = (20, 10)


def display(image, title, cmap=None):
    """Утилита для отображения изображений"""
    plt.figure()
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if cmap is None else image,
               cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()


# 1. Загрузка и препроцессинг изображения
try:
    image = cv2.imread('IMG_cup.jpg')
    if image is None:
        raise FileNotFoundError("Файл не найден!")
except Exception as e:
    print(f"Ошибка загрузки изображения: {e}")
    exit()

# Изменение размера и отображение оригинала
image = cv2.resize(image, (640, 480))  # Фиксированный размер для стабильности
display(image, "Исходное изображение")

# 2. Конвертация в градации серого
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
display(gray, "Черно-белое изображение", cmap='gray')

# 3. Размытие и детекция границ
gray_blurred = cv2.GaussianBlur(gray, (9, 9), 0)
edged = cv2.Canny(gray_blurred, 30, 150)
display(edged, "Границы (Canny)", cmap='gray')

# 4. Поиск контуров
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

if not cnts:
    print("Контуры не найдены!")
    exit()

# 5. Поиск четырехугольника
cnts_sorted = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]  # Топ-10 контуров

quadrilateral = None
for c in cnts_sorted:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    if len(approx) == 4:
        quadrilateral = approx
        print("Найден четырехугольник!")
        break

# Отрисовка четырехугольника
if quadrilateral is not None:
    image_quad = image.copy()
    cv2.drawContours(image_quad, [quadrilateral], -1, (0, 255, 0), 3)
    display(image_quad, "Найденный четырехугольник")
else:
    print("Четырехугольник не обнаружен")

# 6. Поиск круга методом контуров
circle_contour = None
for c in cnts_sorted:
    (x, y), radius = cv2.minEnclosingCircle(c)
    radius = int(radius)
    contour_area = cv2.contourArea(c)
    circle_area = np.pi * (radius ** 2)

    # Параметры можно менять!
    if (radius > 20 and
            contour_area / circle_area > 0.7 and
            abs(cv2.contourArea(c) - circle_area) < 1000):
        circle_contour = c
        print(f"Найден круг: радиус {radius}, площадь {contour_area:.1f}")
        break

# Отрисовка круга через контуры
if circle_contour is not None:
    image_circle = image.copy()
    (x, y), radius = cv2.minEnclosingCircle(circle_contour)
    center = (int(x), int(y))
    cv2.circle(image_circle, center, int(radius), (255, 0, 0), 3)
    cv2.drawContours(image_circle, [circle_contour], -1, (0, 255, 0), 2)
    display(image_circle, "Круг через контуры")
else:
    print("Круг через контуры не найден")

# 7. Альтернативный метод: Transform Hough для кругов
try:
    circles = cv2.HoughCircles(
        gray_blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=50,
        param2=30,
        minRadius=20,
        maxRadius=200
    )

    if circles is not None:
        image_hough = image.copy()
        circles = np.uint16(np.around(circles))

        for i in circles[0, :]:
            cv2.circle(image_hough, (i[0], i[1]), i[2], (0, 0, 255), 3)

        display(image_hough, "Круги через Hough Transform")
    else:
        print("Круги методом Hough не найдены")

except Exception as e:
    print(f"Ошибка в HoughCircles: {e}")

# 8. Финализация: совмещение всех результатов
final_image = image.copy()
if quadrilateral is not None:
    cv2.drawContours(final_image, [quadrilateral], -1, (0, 255, 0), 2)
if circle_contour is not None:
    cv2.drawContours(final_image, [circle_contour], -1, (255, 0, 0), 2)
if circles is not None:
    for i in circles[0, :]:
        cv2.circle(final_image, (i[0], i[1]), i[2], (0, 0, 255), 2)

display(final_image, "Финальный результат")