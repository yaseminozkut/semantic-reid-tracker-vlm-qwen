# detector/yolov8.py

from ultralytics import YOLO
import numpy as np
import cv2

class YOLOv8PersonDetector:
    def __init__(self, model_name="yolov8n.pt", conf_threshold=0.3):
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold

    def detect_persons(self, frame):
        """
        Runs YOLOv8 on the input frame and returns a list of person detections.

        Each detection is a dict:
        {
            "bbox": [x1, y1, x2, y2],
            "confidence": float,
            "class_id": int,
            "class_name": str
        }
        """
        results = self.model.predict(source=frame, conf=self.conf_threshold, classes=[0], verbose=False)  # class 0 = person

        detections = []
        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]

                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf,
                    "class_id": class_id,
                    "class_name": class_name
                })

        return detections
