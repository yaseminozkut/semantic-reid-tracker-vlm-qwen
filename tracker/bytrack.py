# tracker/ultralytics_bytrack.py

from ultralytics import YOLO
import cv2

class UltralyticsByteTrack:
    def __init__(self, model_path="yolov8m.pt", tracker_cfg="bytetrack.yaml", persist=True):
        self.model = YOLO(model_path)
        self.tracker_cfg = tracker_cfg
        self.persist = persist

    def track_frame(self, frame):
        """
        Runs detection and tracking on a single frame.
        Returns a list of dicts, one per person:
        {
            "track_id": int,
            "bbox": [x1, y1, x2, y2],
            "confidence": float,
            "crop": person_img (numpy array)
        }
        """
        results = self.model.track(
            frame,
            persist=self.persist,
            tracker=self.tracker_cfg,
            classes=0,  # only person
            verbose=False
        )
        output = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                box = boxes[i]
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                track_id = int(box.id[0]) if box.id is not None else None
                conf = float(box.conf[0])
                crop = frame[y1:y2, x1:x2].copy() if x2 > x1 and y2 > y1 else None
                output.append({
                    "track_id": track_id,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf,
                    "crop": crop
                })
        return output
