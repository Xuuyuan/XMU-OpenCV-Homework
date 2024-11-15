import cv2
import numpy
raw = cv2.imread("table.png")
gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
binary = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, -5)
rows, cols = binary.shape
scale = 40
# 横线
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cols // scale, 1))
eroded = cv2.erode(binary, kernel, iterations=1)
dilated_col = cv2.dilate(eroded, kernel, iterations=1)
# 竖线
scale = 20
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, rows // scale))
eroded = cv2.erode(binary, kernel, iterations=1)
dilated_row = cv2.dilate(eroded, kernel, iterations=1)
bitwise_and = cv2.bitwise_and(dilated_col, dilated_row)
merge = cv2.add(dilated_col, dilated_row)
merge2 = cv2.subtract(binary, merge)
new_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
erode_image = cv2.morphologyEx(merge2, cv2.MORPH_OPEN, new_kernel)
merge3 = cv2.add(erode_image, bitwise_and)
ys, xs = numpy.where(bitwise_and > 0)
y_point_arr = []
x_point_arr = []
i = 0
sort_x_point = numpy.sort(xs)
for i in range(len(sort_x_point) - 1):
    if sort_x_point[i + 1] - sort_x_point[i] > 10:
        x_point_arr.append(sort_x_point[i])
    i = i + 1
x_point_arr.append(sort_x_point[i])
i = 0
sort_y_point = numpy.sort(ys)
for i in range(len(sort_y_point) - 1):
    if sort_y_point[i + 1] - sort_y_point[i] > 10:
        y_point_arr.append(sort_y_point[i])
    i = i + 1
y_point_arr.append(sort_y_point[i])
num = 1
for k in range(len(y_point_arr) - 1):
    for i in range(len(x_point_arr) - 1):
        print(f"图片{num} 左上{x_point_arr[i]},{y_point_arr[k]} 右下{x_point_arr[i+1]},{y_point_arr[k+1]}")
        cv2.imwrite(f"{num}.png", raw[y_point_arr[k]:y_point_arr[k+1], x_point_arr[i]:x_point_arr[i+1]])
        num += 1
cv2.imshow("pic", raw)
cv2.waitKey(0)
cv2.destroyAllWindows()
