from ultralytics import YOLO
import numpy

#load pretrained model yolov8
model=YOLO("yolov8n.pt","v8")

#predict on an image
detection_output=model.predict(source="/Users/apple/Downloads/archive-2/3.jpg",conf=0.25,save=True)

#display tensor array
print(detection_output)

#display numpy array
print(detection_output[0].numpy())