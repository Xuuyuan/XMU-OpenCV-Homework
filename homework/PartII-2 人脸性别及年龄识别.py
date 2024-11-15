from openvino.inference_engine import IECore
import numpy as np
import time
import cv2 as cv
genders = ['female', 'male']

def demo():
    ie = IECore()
    for device in ie.available_devices:
        print(device)

    model_xml = "./intel/face-detection-0200/FP16/face-detection-0200.xml"
    model_bin = "./intel/face-detection-0200/FP16/face-detection-0200.bin"

    net = ie.read_network(model=model_xml, weights=model_bin)
    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))

    n, c, h, w = net.input_info[input_blob].input_data.shape
    print(n, c, h, w)

    cap = cv.VideoCapture("partii-testvideo1.mp4")
    exec_net = ie.load_network(network=net, device_name="CPU")

    em_xml = "./intel/age-gender-recognition-retail-0013/FP16/age-gender-recognition-retail-0013.xml"
    em_bin = "./intel/age-gender-recognition-retail-0013/FP16/age-gender-recognition-retail-0013.bin"

    em_net = ie.read_network(model=em_xml, weights=em_bin)
    em_input_blob = next(iter(em_net.input_info))
    em_it = iter(em_net.outputs)
    em_out_blob1 = next(em_it)
    em_out_blob2 = next(em_it)
    en, ec, eh, ew = em_net.input_info[em_input_blob].input_data.shape
    print(en, ec, eh, ew)

    em_exec_net = ie.load_network(network=em_net, device_name="CPU")

    while True:
        ret, frame = cap.read()
        if ret is not True:
            break
        image = cv.resize(frame, (w, h))
        image = image.transpose(2, 0, 1)
        inf_start = time.time()
        res = exec_net.infer(inputs={input_blob:[image]})
        inf_end = time.time() - inf_start
        ih, iw, ic = frame.shape
        res = res[out_blob]
        for obj in res[0][0]:
            if obj[2] > 0.75:
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
                roi_img = cv.resize(roi, (ew, eh))
                roi_img = roi_img.transpose(2, 0, 1)
                em_res = em_exec_net.infer(inputs={em_input_blob: [roi_img]})
                age_conv3 = em_res[em_out_blob1].reshape(1, 1)[0][0] * 100
                prob_age = em_res[em_out_blob2].reshape(1, 2)
                label_index = np.int64(np.argmax(prob_age, 1))
                age = np.int64(age_conv3)
                cv.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2, 8)
                cv.putText(frame, np.array(genders)[label_index][0] + ', ' +str(age), (xmin, ymin), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, 8)
        cv.imshow("demo", frame)
        if cv.waitKey(20) & 0xFF == ord("q"):
            break
    cv.destroyAllWindows()

demo()