# src/embedding/clip_embedder.py

import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np

class ClipEmbedder:
    def __init__(self, model_path, device="cuda"):
        self.device = device
        self.model = CLIPModel.from_pretrained(model_path).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_path)

    def get_embedding(self, image):
        # Ensure image is a PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
            outputs = outputs / outputs.norm(dim=-1, keepdim=True)
        return outputs.cpu().numpy().flatten()