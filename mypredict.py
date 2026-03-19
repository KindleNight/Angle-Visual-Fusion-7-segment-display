from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.predict(
    source=r"ultralytics/assets",
    show=False,
    save=True,
    max_det=300,
    # visualize = True,
)