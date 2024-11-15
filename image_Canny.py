import cv2

img = cv2.imread("img.png", 0)
dst = cv2.Canny(img, 150, 200)
cv2.imshow("org image", img)
cv2.imshow("new image", dst)

cv2.waitKey(0)
cv2.destroyAllWindows()
