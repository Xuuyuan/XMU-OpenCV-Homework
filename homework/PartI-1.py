# 利用OpenCV库中图像和视频的相关操作，
# 设计实现摄像头采集视频并直接播放、
# 摄像头采集视频并水平镜像播放、
# 摄像头采集视频并垂直翻转播放。
import cv2 as cv
mode = int(input('输入1进入水平镜像播放，输入2进入垂直翻转播放，输入其它内容直接播放:'))
cap = cv.VideoCapture(0)
# 采集视频并直接播放
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if mode == 1:
        frame = cv.flip(frame, 1) # 水平
    elif mode == 2:
        frame = cv.flip(frame, 0) # 垂直
    cv.imshow("frame", frame)
    if cv.waitKey(20) == ord("q"):
        break
cap.release()
cv.destroyAllWindows()