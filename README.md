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

```text
Video File
   │
   ▼
YOLOv8 Model
   │
   ▼
Crop Frames
   │
   ▼
Qwen2.5-VL-3B-Instruct (VLM)
   │   • Generates structured JSON attributes per person crop
   │
   ▼
Memory Matcher
   │   • with Qwen2.5-7B-Instruct resoning
   │
   ▼
Global ID Assignment
```

---

## 🧠 Example Memory & Reasoning

### Final Memory Store
<details>
<summary>Click to expand the full final memory store (IDs 0–12) to compare descriptions and the ID assignment in the video</summary>

```javascript
  //ID 0:
  {
  "gender": "male",
  "age_group": "adult",
  "hair_color": "black",
  "hair_tone": "jet",
  "hair_length": "short",
  "eyebrow_shape": "thick",
  "jawline": "rounded",
  "face_shape": "square",
  "hair_style": "straight",
  "facial_hair": "none",
  "eyewear": "none",
  "skin_tone": "fair",
  "body_type": "average",
  "distinctive_marks": "none",
  "clothes": "black long-sleeve shirt over a blue dress shirt",
  "bag": "none",
  "gloves": "none",
  "footwear": "none; not visible",
  "headwear": "none",
  "accessories": "none"
}

//ID 1:
{
  "gender": "female",
  "age_group": "adult",
  "hair_color": "brown",
  "hair_tone": "chocolate",
  "hair_length": "shoulder_length",
  "eyebrow_shape": "thick",
  "jawline": "rounded",
  "face_shape": "oval",
  "hair_style": "updo",
  "facial_hair": "none",
  "eyewear": "none",
  "skin_tone": "medium",
  "body_type": "average",
  "distinctive_marks": "none",
  "clothes": "a brown patterned sweater over a white shirt, paired with white pants",
  "bag": "none",
  "gloves": "none",
  "footwear": "none; not_visible",
  "headwear": "none; not_visible",
  "accessories": "a necklace with a red pendant, a watch on her left wrist"
}

//ID 2:
{
  "gender": "female",
  "age_group": "adult",
  "hair_color": "black",
  "hair_tone": "jet",
  "hair_length": "shoulder_length",
  "eyebrow_shape": "thick",
  "jawline": "rounded",
  "face_shape": "oval",
  "hair_style": "bun",
  "facial_hair": "none",
  "eyewear": "none",
  "skin_tone": "medium",
  "body_type": "average",
  "distinctive_marks": "none",
  "clothes": "a brown plaid shirt with a V-neckline, buttoned up to the chest, and a necklace with a small pendant.",
  "bag": "none;",
  "gloves": "none",
  "footwear": "none;",
  "headwear": "none;",
  "accessories": "earrings"
}

//ID 3:
{
  "gender": "male",
  "age_group": "adult",
  "hair_color": "brown",
  "hair_tone": "jet",
  "hair_length": "shoulder_length",
  "eyebrow_shape": "thick",
  "jawline": "rounded",
  "face_shape": "square",
  "hair_style": "straight",
  "facial_hair": "none",
  "eyewear": "prescription_glasses",
  "skin_tone": "fair",
  "body_type": "average",
  "distinctive_marks": "none",
  "clothes": "black velvet coat with multiple buttons, long sleeves, and a high collar",
  "bag": "none",
  "gloves": "none",
  "footwear": "none; not_visible",
  "headwear": "none",
  "accessories": "none"
}

//ID 4:
{
  "gender": "female",
  "age_group": "adult",
  "hair_color": "blonde",
  "hair_tone": "honey",
  "hair_length": "shoulder_length",
  "eyebrow_shape": "thick",
  "jawline": "rounded",
  "face_shape": "oval",
  "hair_style": "straight",
  "facial_hair": "none",
  "eyewear": "none",
  "skin_tone": "fair",
  "body_type": "average",
  "distinctive_marks": "none",
  "clothes": "a black dress with a floral pattern, loose fit",
  "bag": "handbag;black",
  "gloves": "none",
  "footwear": "sneakers;black",
  "headwear": "none",
  "accessories": "none"
}

//ID 5:
{
  "gender": "female",
  "age_group": "adult",
  "hair_color": "black",
  "hair_tone": "jet",
  "hair_length": "shoulder_length",
  "eyebrow_shape": "thick",
  "jawline": "rounded",
  "face_shape": "oval",
  "hair_style": "ponytail",
  "facial_hair": "none",
  "eyewear": "none",
  "skin_tone": "medium",
  "body_type": "average",
  "distinctive_marks": "none",
  "clothes": "a patterned blouse with a deep V-neckline, featuring a mix of earthy tones like brown, orange, and red, with a subtle texture that suggests a woven or knit material.",
  "bag": "none;",
  "gloves": "none",
  "footwear": "none;",
  "headwear": "none;",
  "accessories": "a necklace with a small pendant, earrings, and a bracelet."
}

//ID 6:
{
  "gender": "male",
  "age_group": "adult",
  "hair_color": "brown",
  "hair_tone": "chocolate",
  "hair_length": "short",
  "eyebrow_shape": "thick",
  "jawline": "rounded",
  "face_shape": "square",
  "hair_style": "wavy",
  "facial_hair": "none",
  "eyewear": "none",
  "skin_tone": "medium",
  "body_type": "average",
  "distinctive_marks": "none",
  "clothes": "black jacket with blue collar",
  "bag": "none;",
  "gloves": "none",
  "footwear": "none;",
  "headwear": "none;",
  "accessories": "none"
}

//ID 7:
{
  "gender": "male",
  "age_group": "adult",
  "hair_color": "brown",
  "hair_tone": "jet",
  "hair_length": "shoulder_length",
  "eyebrow_shape": "thick",
  "jawline": "rounded",
  "face_shape": "oval",
  "hair_style": "straight",
  "facial_hair": "none",
  "eyewear": "none",
  "skin_tone": "fair",
  "body_type": "average",
  "distinctive_marks": "none",
  "clothes": "black button-up shirt, black pants",
  "bag": "none",
  "gloves": "none",
  "footwear": "none; not_visible",
  "headwear": "none",
  "accessories": "none"
}

//ID 8:
{
  "gender": "female",
  "age_group": "adult",
  "hair_color": "brown",
  "hair_tone": "chocolate",
  "hair_length": "short",
  "eyebrow_shape": "thick",
  "jawline": "rounded",
  "face_shape": "round",
  "hair_style": "bun",
  "facial_hair": "none",
  "eyewear": "none",
  "skin_tone": "medium",
  "body_type": "average",
  "distinctive_marks": "none",
  "clothes": "a brown plaid shirt with a ruffled collar and buttons down the front",
  "bag": "none",
  "gloves": "none",
  "footwear": "none; not visible",
  "headwear": "none; not visible",
  "accessories": "a pearl necklace with a small pendant"
}

//ID 9:
{
  "gender": "male",
  "age_group": "adult",
  "hair_color": "brown",
  "hair_tone": "chocolate",
  "hair_length": "shoulder_length",
  "eyebrow_shape": "thick",
  "jawline": "rounded",
  "face_shape": "oval",
  "hair_style": "straight",
  "facial_hair": "none",
  "eyewear": "none",
  "skin_tone": "medium",
  "body_type": "average",
  "distinctive_marks": "none",
  "clothes": "a black sweater vest over a light blue button-up shirt, paired with dark pants",
  "bag": "none",
  "gloves": "none",
  "footwear": "none; brown",
  "headwear": "none",
  "accessories": "none"
}

//ID 10:
{
  "gender": "female",
  "age_group": "adult",
  "hair_color": "blonde",
  "hair_tone": "golden",
  "hair_length": "shoulder_length",
  "eyebrow_shape": "thick",
  "jawline": "rounded",
  "face_shape": "oval",
  "hair_style": "updo",
  "facial_hair": "none",
  "eyewear": "none",
  "skin_tone": "medium",
  "body_type": "average",
  "distinctive_marks": "none",
  "clothes": "black shirt with black vest",
  "bag": "none",
  "gloves": "none",
  "footwear": "none; not_visible",
  "headwear": "none",
  "accessories": "striped choker necklace"
}

//ID 11:
{
  "gender": "female",
  "age_group": "adult",
  "hair_color": "brown",
  "hair_tone": "chocolate",
  "hair_length": "short",
  "eyebrow_shape": "thick",
  "jawline": "rounded",
  "face_shape": "round",
  "hair_style": "straight",
  "facial_hair": "none",
  "eyewear": "none",
  "skin_tone": "medium",
  "body_type": "average",
  "distinctive_marks": "none",
  "clothes": "a black and purple striped long-sleeve shirt",
  "bag": "none",
  "gloves": "none",
  "footwear": "none; not visible",
  "headwear": "none; not visible",
  "accessories": "none"
}

//ID 12:
{
  "gender": "female",
  "age_group": "adult",
  "hair_color": "blonde",
  "hair_tone": "golden",
  "hair_length": "shoulder_length",
  "eyebrow_shape": "thick",
  "jawline": "rounded",
  "face_shape": "oval",
  "hair_style": "updo",
  "facial_hair": "none",
  "eyewear": "none",
  "skin_tone": "medium",
  "body_type": "average",
  "distinctive_marks": "none",
  "clothes": "black shirt with a striped choker around the neck",
  "bag": "none",
  "gloves": "none",
  "footwear": "none; not_visible",
  "headwear": "none",
  "accessories": "striped choker"
}
]
```

</details>

### Sample Matching Reasoning

```jsonc
// Frame 0: new person → ID 0
{
  "matched_id": null,
  "confidence": "low",
  "reasoning": "The gender, facial hair, and general appearance are very different. The clothes and accessories have minor similarities but do not outweigh the overall differences."
}

// Later: matching to ID 1
{
  "matched_id": 1,
  "confidence": "high",
  "reasoning": "The new description matches ID 1 in all core physical features including hair color, length, tone, eyebrow shape, jawline, face shape, and skin tone. The clothes and accessories also align closely."
}

// And another match back to ID 0
{
  "matched_id": 0,
  "confidence": "high",
  "reasoning": "The new description closely matches ID 0 in core features like gender, age group, hair color and tone, jawline, face shape, eyebrow shape, skin tone, body type, and lack of facial hair and eyewear. The clothes description is slightly different but does not contradict the core features."
}

```
--- 

## ⚠️ Limitations

- **High Latency:** Per-frame VLM + LLM calls are slow (the unoptimized pipeline can take ~1 hr for a 10 s clip. Even after batching and tracklet summarization, real-time performance remains challenging)
- **Semantic Ambiguity:** Descriptions (“blue shirt”, “short hair”) can be too coarse to distinguish similar appearances (e.g. identical uniforms), leading to ID swaps.
- **Occlusion & Pose Variation:** When people are partially occluded or appear in dramatically different poses, generated attributes may change or become “unknown,” breaking continuity.
- **Prompt Sensitivity:** Small changes in prompt phrasing can yield inconsistent attribute outputs. Requires careful prompt engineering and validation.
- **Memory Growth:** Storing every unique description over long videos can bloat the “memory” file. You may need periodic pruning or clustering to keep it manageable.
---

## 🔮 Future Work

- **Hybrid Embedding + Semantic:** Fuse CLIP-style embeddings with language descriptions.

- **Appearance-change Detection:** Only re-describe when an appearance shift is detected.

- **On-device Inference:** Explore smaller, open-source VLM/LLM for edge deployment.

- **User-guided ReID:** Interactive correction loop to refine memory entries in real time.

## 🧠 Example Memory & Reasoning

<details>
<summary>Click to expand the full final memory store (IDs 0–12)</summary>

```json
[
  {
    "id": 0,
    "gender": "male",
    "age_group": "adult",
    "hair_color": "black",
    "hair_tone": "jet",
    "hair_length": "short",
    "eyebrow_shape": "thick",
    "jawline": "rounded",
    "face_shape": "square",
    "hair_style": "straight",
    "facial_hair": "none",
    "eyewear": "none",
    "skin_tone": "fair",
    "body_type": "average",
    "distinctive_marks": "none",
    "clothes": "black long-sleeve shirt over a blue dress shirt",
    "bag": "none",
    "gloves": "none",
    "footwear": "none; not visible",
    "headwear": "none",
    "accessories": "none"
  },
  {
    "id": 1,
    "gender": "female",
    "age_group": "adult",
    "hair_color": "brown",
    "hair_tone": "chocolate",
    "hair_length": "shoulder_length",
    "eyebrow_shape": "thick",
    "jawline": "rounded",
    "face_shape": "oval",
    "hair_style": "updo",
    "facial_hair": "none",
    "eyewear": "none",
    "skin_tone": "medium",
    "body_type": "average",
    "distinctive_marks": "none",
    "clothes": "a brown patterned sweater over a white shirt, paired with white pants",
    "bag": "none",
    "gloves": "none",
    "footwear": "none; not_visible",
    "headwear": "none; not_visible",
    "accessories": "a necklace with a red pendant, a watch on her left wrist"
  },
  {
    "id": 2,
    "gender": "female",
    "age_group": "adult",
    "hair_color": "black",
    "hair_tone": "jet",
    "hair_length": "shoulder_length",
    "eyebrow_shape": "thick",
    "jawline": "rounded",
    "face_shape": "oval",
    "hair_style": "bun",
    "facial_hair": "none",
    "eyewear": "none",
    "skin_tone": "medium",
    "body_type": "average",
    "distinctive_marks": "none",
    "clothes": "a brown plaid shirt with a V-neckline, buttoned up to the chest, and a necklace with a small pendant.",
    "bag": "none",
    "gloves": "none",
    "footwear": "none",
    "headwear": "none",
    "accessories": "earrings"
  },
  {
    "id": 3,
    "gender": "male",
    "age_group": "adult",
    "hair_color": "brown",
    "hair_tone": "jet",
    "hair_length": "shoulder_length",
    "eyebrow_shape": "thick",
    "jawline": "rounded",
    "face_shape": "square",
    "hair_style": "straight",
    "facial_hair": "none",
    "eyewear": "prescription_glasses",
    "skin_tone": "fair",
    "body_type": "average",
    "distinctive_marks": "none",
    "clothes": "black velvet coat with multiple buttons, long sleeves, and a high collar",
    "bag": "none",
    "gloves": "none",
    "footwear": "none; not_visible",
    "headwear": "none",
    "accessories": "none"
  },
  {
    "id": 4,
    "gender": "female",
    "age_group": "adult",
    "hair_color": "blonde",
    "hair_tone": "honey",
    "hair_length": "shoulder_length",
    "eyebrow_shape": "thick",
    "jawline": "rounded",
    "face_shape": "oval",
    "hair_style": "straight",
    "facial_hair": "none",
    "eyewear": "none",
    "skin_tone": "fair",
    "body_type": "average",
    "distinctive_marks": "none",
    "clothes": "a black dress with a floral pattern, loose fit",
    "bag": "handbag;black",
    "gloves": "none",
    "footwear": "sneakers;black",
    "headwear": "none",
    "accessories": "none"
  },
  {
    "id": 5,
    "gender": "female",
    "age_group": "adult",
    "hair_color": "black",
    "hair_tone": "jet",
    "hair_length": "shoulder_length",
    "eyebrow_shape": "thick",
    "jawline": "rounded",
    "face_shape": "oval",
    "hair_style": "ponytail",
    "facial_hair": "none",
    "eyewear": "none",
    "skin_tone": "medium",
    "body_type": "average",
    "distinctive_marks": "none",
    "clothes": "a patterned blouse with a deep V-neckline, featuring a mix of earthy tones like brown, orange, and red, with a subtle texture that suggests a woven or knit material.",
    "bag": "none",
    "gloves": "none",
    "footwear": "none",
    "headwear": "none",
    "accessories": "a necklace with a small pendant, earrings, and a bracelet."
  },
  {
    "id": 6,
    "gender": "male",
    "age_group": "adult",
    "hair_color": "brown",
    "hair_tone": "chocolate",
    "hair_length": "short",
    "eyebrow_shape": "thick",
    "jawline": "rounded",
    "face_shape": "square",
    "hair_style": "wavy",
    "facial_hair": "none",
    "eyewear": "none",
    "skin_tone": "medium",
    "body_type": "average",
    "distinctive_marks": "none",
    "clothes": "black jacket with blue collar",
    "bag": "none",
    "gloves": "none",
    "footwear": "none",
    "headwear": "none",
    "accessories": "none"
  },
  {
    "id": 7,
    "gender": "male",
    "age_group": "adult",
    "hair_color": "brown",
    "hair_tone": "jet",
    "hair_length": "shoulder_length",
    "eyebrow_shape": "thick",
    "jawline": "rounded",
    "face_shape": "oval",
    "hair_style": "straight",
    "facial_hair": "none",
    "eyewear": "none",
    "skin_tone": "fair",
    "body_type": "average",
    "distinctive_marks": "none",
    "clothes": "black button-up shirt, black pants",
    "bag": "none",
    "gloves": "none",
    "footwear": "none; not_visible",
    "headwear": "none",
    "accessories": "none"
  },
  {
    "id": 8,
    "gender": "female",
    "age_group": "adult",
    "hair_color": "brown",
    "hair_tone": "chocolate",
    "hair_length": "short",
    "eyebrow_shape": "thick",
    "jawline": "rounded",
    "face_shape": "round",
    "hair_style": "bun",
    "facial_hair": "none",
    "eyewear": "none",
    "skin_tone": "medium",
    "body_type": "average",
    "distinctive_marks": "none",
    "clothes": "a brown plaid shirt with a ruffled collar and buttons down the front",
    "bag": "none",
    "gloves": "none",
    "footwear": "none; not visible",
    "headwear": "none; not visible",
    "accessories": "a pearl necklace with a small pendant"
  },
  {
    "id": 9,
    "gender": "male",
    "age_group": "adult",
    "hair_color": "brown",
    "hair_tone": "chocolate",
    "hair_length": "shoulder_length",
    "eyebrow_shape": "thick",
    "jawline": "rounded",
    "face_shape": "oval",
    "hair_style": "straight",
    "facial_hair": "none",
    "eyewear": "none",
    "skin_tone": "medium",
    "body_type": "average",
    "distinctive_marks": "none",
    "clothes": "a black sweater vest over a light blue button-up shirt, paired with dark pants",
    "bag": "none",
    "gloves": "none",
    "footwear": "none; brown",
    "headwear": "none",
    "accessories": "none"
  },
  {
    "id": 10,
    "gender": "female",
    "age_group": "adult",
    "hair_color": "blonde",
    "hair_tone": "golden",
    "hair_length": "shoulder_length",
    "eyebrow_shape": "thick",
    "ja



