# src/pipeline/graph.py

from sre_parse import State
from langgraph.graph import StateGraph, START, END
from src.utils.viz import draw_detections
import cv2
from PIL import Image
import time
import torch, json
import re
import numpy as np

def wrap_text(text, font, font_scale, thickness, max_width):
    """
    Break `text` into a list of lines, none of which (in pixels) exceed max_width
    when rendered with cv2.putText(font, font_scale, thickness).
    """
    words = text.split()
    lines = []
    current = ""
    for w in words:
        test = current + (" " if current else "") + w
        (tw, _), _ = cv2.getTextSize(test, font, font_scale, thickness)
        if tw <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = w
    if current:
        lines.append(current)
    return lines

def extract_json_from_reply(reply):
    # This regex finds all {...} blocks in the reply
    matches = list(re.finditer(r'\{.*?\}', reply, re.DOTALL))
    if matches:
        # Return the last match (the actual answer)
        return matches[-1].group(0)
    else:
        return None

# You may want to import your detector, but pass it in via state for flexibility

def detection_node(state):
    """
    Runs person detection and tracking on the current frame.
    """
    detector = state["detector"]
    conf_thresh = state["config"].detection.confidence_threshold
    detections = detector.track_frame(state["frame"])
    state["detections"] = [d for d in detections if d["confidence"] >= conf_thresh]
    return state

def embedding_node(state):
    """
    Extracts embeddings for each detected person.
    """
    embedder = state["embedder"]
    embeddings = []
    for det in state["detections"]:
        crop = det["crop"]  # Assumes detector saves the crop in each detection
        embedding = embedder.get_embedding(crop)
        embeddings.append(embedding)
    state["embeddings"] = embeddings
    print(f"Frame {state['frame_id']}: {len(embeddings)} embeddings, shapes: {[e.shape for e in embeddings]}")
    return state

def description_node(state):
    descriptor = state["descriptor"]
    detections = state["detections"]
    crops = []
    for det in detections:
        crop = det["crop"]
        if crop is not None and crop.shape[0] > 10 and crop.shape[1] > 10:
            pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            crops.append(pil_crop)
        else:
            crops.append(None)

    # Only pass valid crops to the batch function
    valid_crops = [c for c in crops if c is not None]
    descriptions = ["[Invalid crop]" if c is None else None for c in crops]

    if valid_crops:
        start = time.time()
        batch_descriptions = descriptor.describe_batch(valid_crops)
        end = time.time()
        print(f"[Timing] describe_batch took {end - start:.2f} seconds for {len(valid_crops)} crops")

        idx = 0
        for i, c in enumerate(crops):
            if c is not None:
                description = batch_descriptions[idx]
                descriptions[i] = extract_json_from_reply(description)
                print(f"[Frame {state['frame_id']}] Person {i} description: {descriptions[i]}")
                idx += 1

    state["descriptions"] = descriptions
    return state

def id_assignment_node(state):
    """Assign global IDs to detections based on embedding similarity."""
    memory = state["memory"]
    embeddings = state["embeddings"]
    global_ids = []

    for embedding in embeddings:
        # Try to find a match in memory
        matched_id, score = memory.find_match(embedding)
        
        if matched_id is not None:
            # Found a match
            global_ids.append(matched_id)
            print(f"Frame {state['frame_id']}: Matched embedding with global ID {matched_id} (score: {score:.3f})")
        else:
            # No match found, add new person to memory
            new_id = memory.add_person(embedding)
            global_ids.append(new_id)
            print(f"Frame {state['frame_id']}: Added new person with global ID {new_id}")

    state["global_ids"] = global_ids
    return state

def id_assignment_description_node(state):
    memory = state["memory"]
    matcher = state["description_matcher"]
    descriptions = state["descriptions"]
    frame_matching_details = state["frame_matching_details"]
    global_ids = []

    for i, description in enumerate(descriptions):
        start = time.time()
        matched_id, confidence, reasoning = memory.find_match_by_description(description, matcher)
        end = time.time()
        print(f"[Timing] LLM comparison took {end - start:.2f} seconds for one description")
        print(matched_id, confidence, reasoning)
        if matched_id is not None:
            global_ids.append(matched_id)
            frame_matching_details.append([matched_id, matched_id, confidence, reasoning])
            print(f"[Frame {state['frame_id']}] Person {i} LLM match: {matched_id}, confidence: {confidence}, reasoning: {reasoning}")
        else:
            new_id = memory.add_person(embedding=None, description=description)
            global_ids.append(new_id)
            frame_matching_details.append([new_id, matched_id, confidence, reasoning])
            print(f"Frame {state['frame_id']}: Added new person with global ID {new_id}")
    state["global_ids"] = global_ids
    return state
    
"""
def output_node(state):
    frame = state["frame"].copy()
    detections    = state["detections"]
    global_ids    = state["global_ids"]
    details       = state["frame_matching_details"]  
    interval = 30
    # details should be a list of (gid, confidence, reasoning)

    h, w, _      = frame.shape
    panel_w      = 300
    fps_panel_bg = 30

    # 1) draw boxes & IDs on the frame
    for det, gid in zip(detections, global_ids):
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, f"ID:{gid}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    rebuild = False
    if state["frame_id"] % interval == 0 or details != state.get("last_details"):
        rebuild = True

    # 2) build the right-hand panel
    panel = np.zeros((h, panel_w, 3), dtype=np.uint8) + fps_panel_bg
    
    if rebuild:
        y     = 20
        dy    = 20
        max_text_width = panel_w - 40

        for entry in details:
            gid = entry[0]
            matched = entry[1]
            confidence = entry[2]
            reasoning = entry[3]
            # header line
            cv2.putText(panel, f"GlobalID: {gid}",
                        (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            y += dy

            # matched_id
            line = f"matched_id: {matched}"
            for sub in wrap_text(line,
                                font=cv2.FONT_HERSHEY_SIMPLEX,
                                font_scale=0.5,
                                thickness=1,
                                max_width=max_text_width):
                cv2.putText(panel, sub, (20, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
                y += dy

            # confidence
            line = f"conf: {confidence}"
            for sub in wrap_text(line,
                                font=cv2.FONT_HERSHEY_SIMPLEX,
                                font_scale=0.5,
                                thickness=1,
                                max_width=max_text_width):
                cv2.putText(panel, sub, (20, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
                y += dy

            # reasoning
            line = f"reason: {reasoning}"
            for sub in wrap_text(line,
                                font=cv2.FONT_HERSHEY_SIMPLEX,
                                font_scale=0.5,
                                thickness=1,
                                max_width=max_text_width):
                cv2.putText(panel, sub, (20, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
                y += dy

            # gap before next entry
            y += dy

            # if we run out of vertical space, break early
            if y > h - dy:
                break

    # 3) stitch left + right and store
    combined = cv2.hconcat([frame, panel])
    state["output_frame"] = combined

    # reset or reinitialize details if needed downstream
    state["frame_matching_details"] = []
    return state
"""
def output_node(state):
    frame      = state["frame"].copy()
    detections = state.get("detections", [])
    global_ids = state.get("global_ids", [])

    for det, gid in zip(detections, global_ids):
        x1, y1, x2, y2 = det["bbox"]
        label          = f"GlobalID:{gid}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        text_y = max(0, y1 - 10)
        cv2.putText(
            frame, label, (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2
        )

    state["output_frame"]           = frame
    state["frame_matching_details"] = []    # clear safely as a list
    return state

# Build the LangGraph
"""
graph = StateGraph(dict)
graph.add_node("detection", detection_node)
graph.add_node("embedding", embedding_node)
graph.add_node("id_assignment", id_assignment_node)
graph.add_node("output", output_node)
graph.add_edge(START, "detection")
graph.add_edge("detection", "embedding")
graph.add_edge("embedding", "id_assignment")  
graph.add_edge("id_assignment", "output")     
graph.add_edge("output", END)
pipeline = graph.compile()
"""
graph = StateGraph(dict)
graph.add_node("detection", detection_node)
graph.add_node("description", description_node)
graph.add_node("id_assignment", id_assignment_description_node)
graph.add_node("output", output_node)
graph.add_edge(START, "detection")
graph.add_edge("detection", "description")
graph.add_edge("description", "id_assignment")  
graph.add_edge("id_assignment", "output")     
graph.add_edge("output", END)
pipeline = graph.compile()