#embedding/moondream_embedder.py
from transformers import AutoModelForCausalLM, AutoTokenizer

class MoonDreamEmbedder:
    def __init__(self, model_id="vikhyatk/moondream2", revision="2025-06-21", device="mps"):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, trust_remote_code=True
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

    def describe(self, pil_img, prompt="Describe this person's appearance for re-identification."):
        enc_img = self.model.encode_image(pil_img)
        answer = self.model.answer_question(enc_img, prompt, self.tokenizer, max_new_tokens=256)
        if isinstance(answer, dict):
            # If it is a dict, extract "answer"
            return answer.get("answer", "").strip()
        else:
            # If it is a string, just return it
            return str(answer).strip()
