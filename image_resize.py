import cv2
img = cv2.imread("img.png")
cv2.imshow("org image", img)
img2 = cv2.resize(img, (400, 400), interpolation=cv2.INTER_CUBIC)  # 随意缩放
cv2.imshow("new image", img2)
cv2.waitKey(0)
img3 = cv2.resize(img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)  # 等比例缩放
cv2.imshow("new image 2", img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
