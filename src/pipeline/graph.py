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
graph.add_node("output", output_node)
graph.add_edge(START, "detection")
graph.add_edge("detection", "output")
graph.add_edge("output", END)
pipeline = graph.compile()
