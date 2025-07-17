# main.py
import os
# Set CUDA device to 2 before importing any CUDA-dependent libraries
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import cv2
from src.detection.detector_tracker import UltralyticsByteTrack
from src.embedding.clip_embedder import ClipEmbedder
from src.memory.memory import PersonMemory
from src.embedding.qwen_embedder import QwenEmbedder
from src.agent.orchestration_agent import OrchestrationAgent
from src.pipeline.graph import pipeline
import yaml
from box import Box
from pathlib import Path
from tqdm import tqdm
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config/friends_config.yaml', help='Path to config file')
args = parser.parse_args()

with open("config/friends_config.yaml", "r") as f:
    config = Box(yaml.safe_load(f))

def main():
    # Debug: Print CUDA device information
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    
    # --- Setup ---
    video_path = config.input_video
    output_path = config.output_video
    detector = UltralyticsByteTrack(
        model_path=config.detection.model_path,
        tracker_cfg=config.tracking.tracker_cfg,
        persist=config.tracking.persist,
        device=config.device
    )
    embedder = ClipEmbedder(
        model_path=config.embedding.model_path, 
        device=config.device
    )
    memory = PersonMemory(similarity_threshold=config.embedding.similarity_threshold)
    descriptor = QwenEmbedder(model_id=config.description.vlm_model, device_map=config.device, prompt=config.description.prompt)
    orchestrator = OrchestrationAgent(
        model_name = config.llm.llm_model,
        quant      = getattr(config.llm, "quant", "4bit"),
        max_new_tokens = getattr(config.llm, "max_new_tokens", 120),
        device_map = getattr(config.llm, "device_map", "auto"),
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
                "embedder": embedder,
                "descriptor": descriptor,
                "detections": [],
                "descriptions": [],
                "description_matcher": orchestrator,
                "embeddings": [], 
                "memory": memory,  
                "global_ids": [],  
                "output_frame": None,
                "frame_matching_details": []
            }

            # --- Run Pipeline ---
            result_state = pipeline.invoke(state)
            out.write(result_state["output_frame"])

            # Print current memory after processing this frame
            print("Current memory:")
            for gid, person in memory.memory.items():
                print(f"ID {gid}: {person['description']}")


            frame_id += 1
            pbar.update(1)

    cap.release()
    out.release()
    print(f"Output saved to {output_path}")
    print(f"Total unique persons tracked: {memory.get_memory_size()}")

if __name__ == "__main__":
    main()
