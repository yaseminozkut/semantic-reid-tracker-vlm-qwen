# src/agent/orchestration_agent.py
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
from langchain_community.llms import HuggingFacePipeline
from langchain.schema import HumanMessage, SystemMessage
import torch, json
import re

def extract_json_from_reply(reply):
    # This regex finds all {...} blocks in the reply
    matches = list(re.finditer(r'\{.*?\}', reply, re.DOTALL))
    if matches:
        # Return the last match (the actual answer)
        return matches[-1].group(0)
    else:
        return None

MATCHING_SYSTEM_PROMPT = """\
You are an expert at matching natural-language person descriptions for re-identification.

Return **only** a valid JSON object with the exact keys:
{
  "matched_id": <integer or null>,
  "confidence": "high|medium|low",
  "reasoning": "<≤40 words>"
}

## Task
Given:
  ① A *new* description of one person.
  ② A dictionary of *existing* person descriptions keyed by global-ID.

Decide whether the new person matches **one—and only one—existing ID**. 
Descriptions will rarely be identical; use overall similarity (per the rubric) rather than exact wording.  
If several IDs seem plausible, choose the single best match. If none is credible, return null.

## Decision rubric
1. **Core physical features (highest weight)**  
   • estimated gender / age range  
   • body build / height bracket  
   • skin tone  
   • hair color / length / style  
   • permanent marks (tattoos, scars, piercings)

2. **Semi-permanent features (medium weight)**  
   • glasses, beard, moustache  
   • jewellery that is hard to remove (earrings etc.)

3. **Clothing & accessories (lowest weight but still important)**  
   Clothing can change; use it to break ties but never as the sole reason to link two IDs.

4. The word **‘unknown’** acts as a wildcard that matches anything in the other description.

5. If multiple IDs seem plausible, pick the *single* best match; if none reach a reasonable
   confidence, return **null**.

"""

class OrchestrationAgent:
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct", quant="4bit", max_new_tokens=120):
        if quant == "4bit":
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
        else:          # plain fp16
            bnb_cfg = None

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",          # spreads across visible GPUs / CPU
            torch_dtype="auto",
            trust_remote_code=True,
            quantization_config=bnb_cfg,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        gen_pipe = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=120,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )

        # LangChain wrapper
        self.tokenizer = tokenizer        # we’ll need it in compare_descriptions
        self.llm = HuggingFacePipeline(pipeline=gen_pipe)
        
    def compare_descriptions(self, new_description, existing_descriptions):
        existing_text = "\n".join(
            [f"ID {gid}: {desc}" for gid, desc in existing_descriptions.items()]
        )

        # Build ChatML messages for Qwen
        messages = [
            {"role": "system", "content": MATCHING_SYSTEM_PROMPT},
            {"role": "user", "content": f"New description: {new_description}\n\nExisting descriptions:\n{existing_text}"},
        ]

        # Convert to a single prompt string using Qwen’s chat template
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        response_text = self.llm.invoke(prompt_text)

        # HuggingFacePipeline returns a str; strip off the leading prompt copy
        reply = response_text.strip()

        json_str = extract_json_from_reply(reply)
        if not json_str:
            return None, "error", f"LLM error: No JSON found in reply\nRaw response: {reply}"

        print("my_json:", json_str)

        try:
            result = json.loads(json_str)
            return (
                result.get("matched_id"),
                result.get("confidence"),
                result.get("reasoning"),
            )
        except Exception as e:
            return None, "error", f"LLM error: {e}\nRaw response: {reply}"
