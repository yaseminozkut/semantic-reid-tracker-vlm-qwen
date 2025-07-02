from transformers import CLIPProcessor, CLIPModel
import torch

class CLIPEmbedder:
    def __init__(self, device="mps"):
        self.device = device
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    def embed(self, pil_img):
        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            emb = self.model.get_image_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)  # normalize
        return emb.squeeze().cpu().numpy()  # to numpy array
