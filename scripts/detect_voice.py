import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cv2
from ultralytics import YOLO
from utils.audio_utils import speak_once

model = YOLO("runs/detect/train/weights/best.pt")

cap = cv2.VideoCapture(0)

prev_objects = set()

while True:
    ret, frame = cap.read()

    frame = cv2.resize(frame, (416, 416))

    results = model(frame, imgsz=416, conf=0.5)

    current_objects = set()

    for box in results[0].boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        current_objects.add(label)

    # Detect only NEW objects
    new_objects = current_objects - prev_objects

    # Smart voice messages
    for obj in new_objects:
        if obj == "person":
            speak_once("Person ahead")
        elif obj == "car":
            speak_once("Vehicle nearby")
        else:
            speak_once(obj)

    prev_objects = current_objects

    cv2.imshow("Detection", results[0].plot())

    if cv2.waitKey(1) == 27:
        break