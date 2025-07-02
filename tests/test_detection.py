# test_bytetrack.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import cv2
from src.detection.detector_tracker import UltralyticsByteTrack
import torch
from tqdm import tqdm

print(torch.cuda.is_available())  # Should print True
print(torch.cuda.device_count())  # Should be >0 if GPU found

# Path to your test video
video_path = "data/videos/friends4.mov"   # CHANGE TO YOUR VIDEO
output_path = "output/tracked_with_crops.mp4"

detector = UltralyticsByteTrack(model_path="yolov8m.pt", device="cuda")

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*"H264")
out_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

frame_idx = 0
with tqdm(total=total_frames, desc="Processing frames") as pbar:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        people = detector.track_frame(frame)

        for person in people:
            x1, y1, x2, y2 = person["bbox"]
            track_id = person["track_id"]
            conf = person["confidence"]
            crop = person["crop"]

            # Draw bbox and ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"ID:{track_id} Conf:{conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        out_writer.write(frame)

        frame_idx += 1
        pbar.update(1)

cap.release()
out_writer.release()
