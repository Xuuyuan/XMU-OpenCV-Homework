import cv2
import numpy as np
img = np.zeros((500, 500, 3), dtype=np.uint8)
color = (37, 63, 110)
cv2.rectangle(img, (10, 10), (410, 460), color, 2)
for i in range(0, 8):
    for k in range(0, 4):
        cv2.rectangle(img, (10+i*50, 10+k*50), (60+i*50, 60+k*50), color, 2)

for i in range(0, 8):
    for k in range(0, 4):
        cv2.rectangle(img, (10+i*50, 260+k*50), (60+i*50, 310+k*50), color, 2)
cv2.line(img, (160, 10), (260, 110), color, 2)
cv2.line(img, (160, 110), (260, 10), color, 2)
cv2.line(img, (160, 360), (260, 460), color, 2)
cv2.line(img, (160, 460), (260, 360), color, 2)
cv2.putText(img, "CUHE", (30, 245), cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness=4)
cv2.putText(img, "HANJIE", (280, 245), cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness=4)
cv2.imshow('canva', img)
cv2.waitKey(0)
cv2.destroyAllWindows()