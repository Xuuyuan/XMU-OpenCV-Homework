import cv2 as cv
import numpy as np
from openvino.inference_engine import IECore

weight_pb = "./cv_model/opencv_face_detector_uint8.pb"
config_text = "./cv_model/opencv_face_detector.pbtxt"

model_xml = r"./intel/emotions-recognition-retail-0003.xml"
model_bin = r"./intel/emotions-recognition-retail-0003.bin"

labels = ['neutral', 'happy', 'sad', 'surprise', 'anger']

ie = IECore()
emotion_net = ie.read_network(model=model_xml, weights=model_bin)

input_blob = next(iter(emotion_net.input_info))

exec_net = ie.load_network(network=emotion_net, device_name="CPU", num_requests=2)
net = cv.dnn.readNetFromTensorflow(weight_pb, config=config_text)
def emotion_detect(frame):
    h, w, c = frame.shape
    blobImage = cv.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)
    net.setInput(blobImage)
    cvOut = net.forward()

    for detection in cvOut[0, 0, :, :]:
        score = float(detection[2])
        if score > 0.5:
            left = detection[3] * w
            top = detection[4] * h
            right = detection[5] * w
            bottom = detection[6] * h

            y1 = np.int32(top) if np.int32(top) > 0 else 0
            y2 = np.int32(bottom) if np.int32(bottom) < h else h - 1
            x1 = np.int32(left) if np.int32(left) > 0 else 0
            x2 = np.int32(right) if np.int32(right) < w else w - 1
            roi = frame[y1:y2, x1:x2, :]
            image = cv.resize(roi, (64, 64))
            image = image.transpose((2, 0, 1))
            res = exec_net.infer(inputs={input_blob: [image]})
            prob_emotion = res['prob_emotion']
            probs = np.reshape(prob_emotion, 5)
            txt = labels[np.argmax(probs)]
            cv.putText(frame, txt, (np.int32(left), np.int32(top)), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
            cv.rectangle(frame, (np.int32(left), np.int32(top)),
                         (np.int32(right), np.int32(bottom)), (0, 0, 255), 2, 8, 0)
capture = cv.VideoCapture('partii-testvideo1.mp4')
while True:
    ret, frame = capture.read()
    if ret is not True:
        break
    emotion_detect(frame)
    cv.imshow("demo", frame)
    if cv.waitKey(20) & 0xFF == ord('q'):
        break