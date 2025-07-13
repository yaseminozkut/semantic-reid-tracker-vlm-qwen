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
