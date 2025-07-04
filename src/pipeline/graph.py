# src/pipeline/graph.py

from sre_parse import State
from langgraph.graph import StateGraph, START, END
from src.utils.viz import draw_detections

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

def output_node(state):
    """
    Draws detections and track IDs on the frame, stores result in 'output_frame'.
    """
    frame = state["frame"].copy()
    state["output_frame"] = draw_detections(frame, state["detections"])
    return state

# Build the LangGraph
graph = StateGraph(dict)
graph.add_node("detection", detection_node)
graph.add_node("embedding", embedding_node)
graph.add_node("output", output_node)
graph.add_edge(START, "detection")
graph.add_edge("detection", "embedding")
graph.add_edge("embedding", "output")
graph.add_edge("output", END)
pipeline = graph.compile()
