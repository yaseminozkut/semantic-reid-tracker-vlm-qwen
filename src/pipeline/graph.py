# src/pipeline/graph.py

from sre_parse import State
from langgraph.graph import StateGraph, START, END
from src.utils.viz import draw_detections
import cv2
from PIL import Image
import time

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
"""   
def description_node(state):
    moondream = state["moondream"]
    descriptions = []
    for det in state["detections"]:
        crop = det["crop"]
        if crop is not None and crop.shape[0] > 10 and crop.shape[1] > 10:
            pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            description = moondream.describe(pil_crop)
            descriptions.append(description)
        else:
            descriptions.append("[Invalid crop]")
    state["descriptions"] = descriptions
    return state
"""
def description_node(state):
    moondream = state["moondream"]
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
        batch_descriptions = moondream.describe_batch(valid_crops)
        end = time.time()
        print(f"[Timing] describe_batch took {end - start:.2f} seconds for {len(valid_crops)} crops")

        idx = 0
        for i, c in enumerate(crops):
            if c is not None:
                descriptions[i] = batch_descriptions[idx]
                if state['frame_id'] <= 1:
                    print(f"[Frame {state['frame_id']}] Person {i} description: {batch_descriptions[idx]}")
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
    global_ids = []
    confidences = []
    reasoning_logs = []

    for i, description in enumerate(descriptions):
        start = time.time()
        matched_id, confidence, reasoning = memory.find_match_by_description(description, matcher)
        end = time.time()
        print(f"[Timing] LLM comparison took {end - start:.2f} seconds for one description")
        print(matched_id, confidence, reasoning)
        if matched_id is not None:
            global_ids.append(matched_id)
            if state['frame_id'] <= 1:
                print(f"[Frame {state['frame_id']}] Person {i} LLM match: {matched_id}, confidence: {confidence}, reasoning: {reasoning}")
        else:
            new_id = memory.add_person(embedding=None, description=description)
            global_ids.append(new_id)
            if state['frame_id'] <= 1:
                print(f"Frame {state['frame_id']}: Added new person with global ID {new_id}")
        confidences.append(confidence)
        reasoning_logs.append(reasoning)
    state["global_ids"] = global_ids
    state["confidences"] = confidences
    state["reasoning_logs"] = reasoning_logs
    return state

def output_node(state):
    """
    Draws detections and global IDs on the frame.
    """
    frame = state["frame"].copy()
    detections = state["detections"]
    global_ids = state["global_ids"]
    confidences = state.get("confidences", [])
    reasonings = state.get("reasoning_logs", [])

    # Draw global IDs on the frame
    for i, (det, gid) in enumerate(zip(detections, global_ids)):
        x1, y1, x2, y2 = det["bbox"]
        label = f"GlobalID:{gid}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # Draw confidence and reasoning
        if i < len(confidences):
            cv2.putText(frame, f"Conf: {confidences[i]}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
        if i < len(reasonings):
            # Optionally truncate reasoning to fit
            short_reason = reasonings[i][:50]
            cv2.putText(frame, short_reason, (x1, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 255), 1)
    
    state["output_frame"] = frame
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