import cv2
img = cv2.imread("img.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", gray)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow("hsv", hsv)
cv2.waitKey(0)
cv2.destroyAllWindows()
