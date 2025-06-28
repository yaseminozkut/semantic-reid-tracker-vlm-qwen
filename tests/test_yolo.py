# test_yolo.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
from detector.yolov8 import YOLOv8PersonDetector

image_path = "data/sample_yolo.jpg"
image = cv2.imread(image_path)

detector = YOLOv8PersonDetector()
detections = detector.detect_persons(image)

for det in detections:
    x1, y1, x2, y2 = det["bbox"]
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, f'{det["class_name"]} {det["confidence"]:.2f}', (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

cv2.imshow("Detections", image)
cv2.waitKey(0)
cv2.destroyAllWindows()