from openvino.inference_engine import IECore
ie = IECore()
print(ie.available_devices)