import cv2

# 待检测的图片路径
cap = cv2.VideoCapture(0)
t = 0
face_cascade = cv2.CascadeClassifier(f'.\haarcascade_frontalface_default.xml')
body_cascade = cv2.CascadeClassifier(f'.\haarcascade_fullbody.xml')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    bodies = body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0 and len(bodies) == 0:
        t += 1
        if t >= 100:
            print("值班人员离岗（超过10s未检测到人脸及人体）")
    else:
        t = 0
        for x, y, width, height in faces:
            cv2.rectangle(frame, (x, y), (x + width, y + height), color=(255, 0, 0), thickness=2)
        for x, y, width, height in bodies:
            cv2.rectangle(frame, (x, y), (x + width, y + height), color=(0, 255, 0), thickness=2)
    cv2.imshow("frame", frame)
    if cv2.waitKey(100) == ord("q"):
        break