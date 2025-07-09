#embedding/moondream_embedder.py
from transformers import AutoModelForCausalLM, AutoTokenizer

prompt = ("""
You’ll see 4 questions about this person’s appearance.  
Answer each in order with a single value (no extra prose).  
Use “not visible” if the item is absent, “unknown” if you can’t tell.

Q1. Is this person male or female?  
Q2. Estimate the age group: child (<18), young adult (18–35), middle-aged (35–60), or senior (>60)?  
Q3. What is the person’s hair color?  
Q4. What is the person’s hair length?

Please output as:
A1: <answer>  
A2: <answer>  
A3: <answer>  
A4: <answer>  
"""
)


class MoonDreamEmbedder:
    def __init__(self, model_id="vikhyatk/moondream2", revision="2025-06-21", device="cuda"):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, trust_remote_code=True
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

    def describe(self, pil_img, prompt=prompt):
        enc_img = self.model.encode_image(pil_img)
        answer = self.model.answer_question(enc_img, prompt, self.tokenizer, max_new_tokens=256, stop_sequences=["A4:"])
        if isinstance(answer, dict):
            # If it is a dict, extract "answer"
            return answer.get("answer", "").strip()
        else:
            # If it is a string, just return it
            return str(answer).strip()
    
    def describe_batch(self, pil_imgs, prompt=prompt):
        # pil_imgs: list of PIL Images
        # prompt: optional, use default if None
        return [self.describe(img, prompt=prompt) for img in pil_imgs]