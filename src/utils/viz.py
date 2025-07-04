# src/utils/viz.py

import cv2

def draw_detections(frame, detections):
    """
    Draws bounding boxes and track IDs on the frame.
    """
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        track_id = det["track_id"]
        conf = det["confidence"]
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"ID:{track_id} {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame