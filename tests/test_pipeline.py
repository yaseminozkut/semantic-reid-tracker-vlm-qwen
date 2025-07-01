import cv2
from PIL import Image
import numpy as np
from tracker.bytrack import UltralyticsByteTrack
from embedding.moondream_embedder import MoonDreamEmbedder
from embedding.clip_embedder import  CLIPEmbedder

# Initialize
tracker = UltralyticsByteTrack(model_path="yolov8m.pt")
moondream = MoonDreamEmbedder()
clipper = CLIPEmbedder()

image_paths = [
    "data/friends1.png",
    "data/friends2.png",
    "data/friends3.png",
    "data/friends4.png"
]
results = []

for img_path in image_paths:
    frame = cv2.imread(img_path)
    people = tracker.track_frame(frame)
    for person in people:
        crop = person["crop"]
        if crop is None:
            continue
        pil_img = Image.fromarray(crop[..., ::-1])
        desc = moondream.describe(pil_img)
        clip_emb = clipper.embed(pil_img)
        results.append({
            "img_path": img_path,
            "desc": desc,
            "clip_emb": clip_emb
        })
        print(f"Image: {img_path}\nDescription: {desc}\n")

n = len(results)
sim_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        sim_matrix[i, j] = cosine_similarity([results[i]["clip_emb"]], [results[j]["clip_emb"]])[0,0]

print("Cosine Similarity Matrix between images:")
print(sim_matrix)
