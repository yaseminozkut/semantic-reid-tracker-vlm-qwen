import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

prompt = """
You are a vision-language assistant. You will be shown an image of a person and asked to fill in attributes.  
– Answer with *only* the values (no extra words, no punctuation beyond JSON syntax).  
– If you can’t see an attribute to reason, reply “not_visible”; if it’s unclear, reply “unknown”; if you can see the attribute but it’s simply absent, reply "none".  
– Conform *exactly* to the JSON form below.
- For clothes field, answer very descriptively

- For bag and footwear field, output exactly four values **in this order**:
    1. type  
    2. color  
  and separate them with semicolons (`;`).

Output *exactly* in this JSON form:
{
  "gender": "<male|female>",
  "age_group": "<baby|child|teen|adult|senior>",
  "hair_color": "<black|brown|brunette|blonde|red|gray|other>",
  "hair_length": "<bald|buzzcut|very short|short|ear_length|chin_length|shoulder_length|medium|long>",
  "hair_style": "<straight|wavy|curly|braided|ponytail|bun|updo|half_up|other>",
  "facial_hair": "<none|mustache|beard|goatee|stubble|moustache_and_beard|other>",
  "eyewear": "<none|prescription_glasses|sunglasses|other>",
  "skin_tone": "<very_fair|fair|light|medium|olive|tan|brown|dark|other>",
  "body_type": "<underweight|slim|average|athletic|overweight|obese|other>",
  "distinctive_marks": "<any valid description e.g. 'tattoo on left forearm', 'scar on right cheek', 'birthmark on neck', 'none'>",
  "clothes": "<describe person's clothes with type, color, pattern, fit information>
  "bag": "<any valid bag type>;<any valid color>",
  "gloves": "<none|gloves|fingerless_gloves|other>",
  "footwear": "<none|sneakers|boots|sandals|heels|flats|loafers|other>;<any valid color>",
  "headwear": "<none|hat|cap|beanie|hood|helmet|other>;<any valid color>",
  "accessories": "<comma-separated list of all visible accessories or 'none'>"
}
"""


class QwenEmbedder:
    def __init__(self, model_id = "Qwen/Qwen2.5-VL-3B-Instruct",
                 torch_dtype = torch.bfloat16, attn_implementation="flash_attention_2",
                 device_map = "cuda"):
        """
        Vision-language embedder using Qwen2.5-VL-3B-Instruct.
        Requires `pip install qwen-vl-utils[decord]` and recent `transformers`.
        """
        # Load the multimodal VL model
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            device_map=device_map,
        )
        # Processor handles tokenization (via its built‑in tokenizer), decoding, and vision preprocessing; no separate tokenizer needed
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.device=device_map

    def describe(self, pil_img, prompt=prompt, max_new_tokens=256):
        """
        Describe a single PIL image given a text prompt.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": pil_img,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        # Prepare vision inputs
        image_inputs, video_inputs = process_vision_info(messages)

        text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        answer = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        return answer.strip()


    def describe_batch(self, pil_imgs, prompt=prompt, max_new_tokens=256):
        """
        Describe a batch of PIL images with the same prompt.
        """
        return [self.describe(img, prompt, max_new_tokens) for img in pil_imgs]
