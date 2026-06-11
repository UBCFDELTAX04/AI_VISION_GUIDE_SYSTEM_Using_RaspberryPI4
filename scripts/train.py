from ultralytics import YOLO

def train():
    model = YOLO("yolov8n.pt")  # nano model (Pi friendly)

    model.train(
        data="config/dataset.yaml",
        epochs=30,
        imgsz=416,
        batch=8,
        workers=2,
        device="cpu"
    )

if __name__ == "__main__":
    train()