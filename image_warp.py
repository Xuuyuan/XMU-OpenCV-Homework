import cv2
img = cv2.imread("img.png")
rows, cols = img.shape[:2]
M = cv2.getRotationMatrix2D((cols/2, rows/2),90,1)
dst = cv2.warpAffine(img, M, (cols, rows))
cv2.imshow('new image', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
