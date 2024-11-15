from openvino.inference_engine import IECore
import numpy as np
import time
import cv2 as cv

def demo():
    ie = IECore()
    for device in ie.available_devices:
        print(device)

    model_xml = ".\intel\person-vehicle-bike-detection-2000.xml"
    model_bin = ".\intel\person-vehicle-bike-detection-2000.bin"

    net = ie.read_network(model=model_xml, weights=model_bin)
    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))

    n, c, h, w = net.input_info[input_blob].input_data.shape
    print(n, c, h, w)

    cap = cv.VideoCapture("partii-testvideo2.mp4")
    exec_net = ie.load_network(network=net, device_name="CPU")

    while True:
        ret, frame = cap.read()
        if ret is not True:
            break
        image = cv.resize(frame, (w, h))
        image = image.transpose(2, 0, 1)
        res = exec_net.infer(inputs={input_blob:[image]})
        ih, iw, ic = frame.shape
        res = res[out_blob]
        num = 0
        for obj in res[0][0]:
            if obj[2] > 0.5:
                xmin = int(obj[3] * iw)
                ymin = int(obj[4] * ih)
                xmax = int(obj[5] * iw)
                ymax = int(obj[6] * ih)
                if xmin < 0:
                    xmin = 0
                if ymin < 0:
                    ymin = 0
                if xmax >= iw:
                    xmax = iw - 1
                if ymax >= ih:
                    ymax = ih - 1
                roi = frame[ymin:ymax,xmin:xmax,:]
                cv.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2, 8)
                num += 1
        cv.putText(frame, str(num), (50,50), cv.FONT_HERSHEY_SIMPLEX, 2, (225,0,0))
        cv.imshow("demo", frame)
        if cv.waitKey(10) & 0xFF == ord("q"):
            break
    cv.destroyAllWindows()

demo()