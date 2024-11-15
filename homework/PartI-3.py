import cv2 as cv, cv2
cap = cv.VideoCapture("ad.mp4")
ad = cv.VideoCapture("video.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    ret2, frame2 = ad.read()
    if not ret2:
        break
    rows, cols, channels = frame2.shape
    frame[10:int(rows*0.6)+10, 10:int(cols*0.6)+10] = cv.resize(frame2, (0, 0), fx=0.6, fy=0.6, interpolation=cv2.INTER_NEAREST)
    cv.imshow("frame", frame)
    if cv.waitKey(20) == ord("q"):
        break
cap.release()
cv.destroyAllWindows()
