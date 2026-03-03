import numpy as np
from ultralytics import YOLO


class YoloPersonDetector:
    def __init__(self, model_path: str = "yolov8n.pt", conf: float = 0.5, iou: float = 0.5):
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.person_class_id = 0  # COCO person

    def detect(self, frame) -> tuple[np.ndarray, np.ndarray]:
        r = self.model.predict(frame, conf=self.conf, iou=self.iou, verbose=False)[0]

        if r.boxes is None or len(r.boxes) == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

        cls = r.boxes.cls.detach().cpu().numpy().astype(int)
        keep = (cls == self.person_class_id)

        xyxy = r.boxes.xyxy.detach().cpu().numpy().astype(np.float32)[keep]
        conf = r.boxes.conf.detach().cpu().numpy().astype(np.float32)[keep]

        if len(xyxy) == 0:
            return xyxy, conf

        # ---- AREA FILTER (INSIDE FUNCTION) ----
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        keep_large = areas > 1000
        xyxy = xyxy[keep_large]
        conf = conf[keep_large]

        return xyxy, conf