from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

# Export to TFLite
model.export(format="tflite")