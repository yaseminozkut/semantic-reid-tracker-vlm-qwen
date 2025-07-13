# Semantic Person Re-Identification via Vision-Language Reasoning
> A proof-of-concept pipeline that uses only YOLOv8 for detection and Qwen2.5-VL-3B-Instruct + Qwen2.5-7B-Instruct for per-person semantic descriptions and matching.

---

## 🚀 Project Overview

Traditional person re-identification relies on specialized embedding networks and metric learning. Here, I explore whether state-of-the-art vision-language (VLM) and language (LLM) models can, by themselves, perform re-id purely via semantic descriptions:

1. **Detect** each person in every video frame with **YOLOv8**.  
2. **Describe** each cropped person via **Qwen2.5-VL-3B-Instruct** into structured JSON attributes (e.g., hair style, clothing, face shape).  
3. **Match** new descriptions against a memory of past descriptions via **Qwen2.5-7B-Instruct**.  
4. **Assign** and maintain **Global IDs** for tracklets based solely on language reasoning.

---

## 🔍 Motivation

- **Explainability**: Semantic attributes (“red jacket, black hat”) are human-readable vs. opaque embedding vectors.  
- **Flexibility**: No need to train a specialized ReID network—leverage existing multimodal APIs.  
- **Exploratory Research**: Push the limits of current VLM/LLM performance on dense video tasks.

---

## ✨ Features

- 🚨 **Real-time detection** with YOLOv8  
- 📝 **Structured JSON** attribute outputs per person crop  
- 🧠 **LLM-driven matching** for global ID assignment  
- ⚙️ **Configurable optimizations**: tracklet summarization, batching, hierarchical matching

---

## 📦 Installation

```bash
git clone https://github.com/yaseminozkut/semantic-reid-tracker-vlm-qwen.git
cd semantic-reid-tracker-vlm-qwen
conda create -n semantic-reid-env python=3.10 -y
conda activate semantic-reid-env
pip install -r requirements.txt
```

---

## 🏗 Architecture

┌────────────┐       ┌──────────────┐       ┌─────────────┐
│ Video File │ ───▶  │ YOLOv8 Model │ ───▶  │ Crop Frames │
└────────────┘       └──────────────┘       └─────────────┘
                                                  │
                                                  ▼
                                         ┌──────────────────────┐
                                         │ Qwen2.5-VL (VLM) API │
                                         │ → Structured JSON    │
                                         └──────────────────────┘
                                                  │
                                                  ▼
                                         ┌──────────────────────┐
                                         │ Memory Matcher:      │
                                         │ → LLM reasoning      │
                                         └──────────────────────┘
                                                  │
                                                  ▼
                                         ┌──────────────────────┐
                                         │ Global ID Assignment │
                                         └──────────────────────┘
---

##⚠️ Limitations

- **High Latency:** Per-frame VLM + LLM calls are slow (the unoptimized pipeline can take ~1 hr for a 10 s clip. Even after batching and tracklet summarization, real-time performance remains challenging)
- **Semantic Ambiguity:** Descriptions (“blue shirt”, “short hair”) can be too coarse to distinguish similar appearances (e.g. identical uniforms), leading to ID swaps.
- **Occlusion & Pose Variation:** When people are partially occluded or appear in dramatically different poses, generated attributes may change or become “unknown,” breaking continuity.
- **Prompt Sensitivity:** Small changes in prompt phrasing can yield inconsistent attribute outputs. Requires careful prompt engineering and validation.
- **Memory Growth:** Storing every unique description over long videos can bloat the “memory” file. You may need periodic pruning or clustering to keep it manageable.
---

##🔮 Future Work

- **Hybrid Embedding + Semantic:** Fuse CLIP-style embeddings with language descriptions.

- **Appearance-change Detection:** Only re-describe when an appearance shift is detected.

- **On-device Inference:** Explore smaller, open-source VLM/LLM for edge deployment.

- **User-guided ReID:** Interactive correction loop to refine memory entries in real time.




