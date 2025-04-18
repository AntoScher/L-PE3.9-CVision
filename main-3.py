import cv2
import numpy as np
from matplotlib import pyplot as plt

# ========== НАСТРОЙКИ ==========
DEBUG = True  # Показывать промежуточные этапы
MIN_QUAD_AREA = 1000  # Минимальная площадь четырехугольника
MIN_CIRCLE_AREA = 300  # Минимальная площадь круга


# ===============================

def debug_show(image, title, cmap=None):
    if DEBUG:
        plt.figure()
        plt.imshow(image if cmap else cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                   cmap=cmap)
        plt.title(title)
        plt.axis('off')
        plt.show()


# Загрузка изображения
image = cv2.imread('IMG_cup.jpg')
image = cv2.resize(image, (640, 480))
debug_show(image, "Original")

# 1. Улучшенная предобработка
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 7)  # Медианный фильтр против шума
debug_show(gray, "Blurred Gray", 'gray')

# 2. Адаптивное пороговое преобразование
thresh = cv2.adaptiveThreshold(
    gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV, 21, 5
)
debug_show(thresh, "Adaptive Threshold", 'gray')

# 3. Морфологические операции
kernel = np.ones((3, 3), np.uint8)
processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel, iterations=1)
debug_show(processed, "Morphological", 'gray')

# 4. Поиск четырехугольника (новый подход)
contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
quadrilaterals = []

for cnt in contours:
    if cv2.contourArea(cnt) < MIN_QUAD_AREA:
        continue

    # Улучшенная аппроксимация
    epsilon = 0.03 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    # Проверка выпуклости
    if len(approx) == 4 and cv2.isContourConvex(approx):
        quadrilaterals.append(approx)

# Выбор самого большого четырехугольника
if quadrilaterals:
    quad = sorted(quadrilaterals, key=cv2.contourArea, reverse=True)[0]
    cv2.drawContours(image, [quad], -1, (0, 255, 0), 3)
else:
    print("Четырехугольники не найдены")

# 5. Детекция круга через эллипсы
circles = []
for cnt in contours:
    if len(cnt) < 5 or cv2.contourArea(cnt) < MIN_CIRCLE_AREA:
        continue

    # Фитируем эллипс
    ellipse = cv2.fitEllipse(cnt)
    (center, axes, angle) = ellipse
    major = max(axes)
    minor = min(axes)

    # Критерий круговости
    if 0.9 < (minor / major) < 1.1 and angle < 20:
        circles.append(ellipse)

# Отрисовка лучшего круга
if circles:
    best_circle = sorted(circles, key=lambda x: x[1][0] * x[1][1], reverse=True)[0]
    cv2.ellipse(image, best_circle, (255, 0, 0), 3)
else:
    print("Круги не найдены")

# 6. Оптимизированный Hough Transform
if not circles:
    hough_circles = cv2.HoughCircles(
        processed,
        cv2.HOUGH_GRADIENT,
        dp=1.5,
        minDist=100,
        param1=200,
        param2=25,
        minRadius=30,
        maxRadius=150
    )

    if hough_circles is not None:
        hough_circles = np.uint16(np.around(hough_circles))
        for c in hough_circles[0, :]:
            cv2.circle(image, (c[0], c[1]), c[2], (0, 0, 255), 3)

# Финальный результат
debug_show(image, "Final Detection")
plt.imsave('result.jpg', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
print("Результат сохранен в result.jpg")