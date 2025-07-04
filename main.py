# main.py

import cv2
from src.detection.detector_tracker import UltralyticsByteTrack
from src.pipeline.graph import pipeline
import yaml
from box import Box
from pathlib import Path
from tqdm import tqdm

with open("config/config.yaml", "r") as f:
    config = Box(yaml.safe_load(f))

def main():
    # --- Setup ---
    video_path = config.input_video
    output_path = config.output_video
    detector = UltralyticsByteTrack(
        model_path=config.detection.model_path,
        tracker_cfg=config.tracking.tracker_cfg,
        persist=config.tracking.persist,
        device=config.device
    )

    # --- Video IO ---
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_id = 0
    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # --- Pipeline State ---
            state = {
                "config": config,
                "frame_id": frame_id,
                "frame": frame,
                "detector": detector,
                "detections": [],
                "track_id_to_global_id": {},
                "output_frame": None,
            }

            # --- Run Pipeline ---
            result_state = pipeline.invoke(state)
            out.write(result_state["output_frame"])

            frame_id += 1
            pbar.update(1)

    cap.release()
    out.release()
    print(f"Output saved to {output_path}")

if __name__ == "__main__":
    main()