from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8-test.yaml")

    model.train(
        data="7-segment-display.yaml",
        epochs=300,
        imgsz=640,
        # batch=64,
        batch=32,
        cache="ram",
        workers=2,
        lr0=0.01,
    )