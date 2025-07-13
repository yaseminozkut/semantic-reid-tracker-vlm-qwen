# tests/test_bytetrack_save.py
import os, sys
from tqdm import tqdm
# go up one level from tests/ into the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.detection.detector_tracker import UltralyticsByteTrack
import cv2
import os

# Input & output paths
input_path = "data/videos/airport.mov"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "vis_airport2.mov")

# Open input
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise IOError(f"Cannot open video {input_path}")

# Video properties
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps    = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define codec and create VideoWriter
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # or 'XVID', 'H264' if available
writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Initialize tracker
tracker = UltralyticsByteTrack(model_path="yolov8m.pt")
with tqdm(total=total_frames, desc="Tracking frames") as pbar:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run tracking
        people = tracker.track_frame(frame)

        # Draw boxes & IDs
        for person in people:
            x1, y1, x2, y2 = person["bbox"]
            track_id       = person["track_id"]
            conf           = person["confidence"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(
                frame,
                f"ID {track_id} ({conf:.2f})",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,255,0),
                2
            )

        # Write annotated frame to output video
        writer.write(frame)
        pbar.update(1)

# Clean up
cap.release()
writer.release()
print(f"Saved tracked video to {output_path}")
