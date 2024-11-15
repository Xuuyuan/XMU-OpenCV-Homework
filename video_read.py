import cv2 as cv
cap = cv.VideoCapture("video.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    cv.imshow("frame", frame)
    if cv.waitKey(20) == ord("q"):
        break
cap.release()
cv.destroyAllWindows()
