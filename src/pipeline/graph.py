# src/pipeline/graph.py

from sre_parse import State
from langgraph.graph import StateGraph, START, END
from src.utils.viz import draw_detections
import cv2

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

def output_node(state):
    """
    Draws detections and global IDs on the frame.
    """
    frame = state["frame"].copy()
    detections = state["detections"]
    global_ids = state["global_ids"]
    
    # Draw global IDs on the frame
    for det, gid in zip(detections, global_ids):
        x1, y1, x2, y2 = det["bbox"]
        label = f"GlobalID:{gid}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    state["output_frame"] = frame
    return state

# Build the LangGraph
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
