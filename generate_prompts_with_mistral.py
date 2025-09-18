import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load mistral model (already downloaded in your setup)
model_id = "mistralai/Mistral-7B-Instruct-v0.1"
pipe = pipeline("text-generation", model=model_id, tokenizer=model_id, device=0)

def get_adjectives(item: str) -> list[str]:
    #prompt = f"Give me 5 descriptive adjectives for a broken or defective {item}."
    # More specific prompt to get better adjectives
    prompt = f"List 5 types of possible defects/anomalies in a {item}. Only return comma-separated words. "
    output = pipe(prompt, max_new_tokens=50, do_sample=True, temperature=0.9)[0]["generated_text"]
    # crude parse: assume adjectives are comma or newline separated
    adjectives = []
    for line in output.split("\n"):
        if ":" in line:
            _, rest = line.split(":", 1)
        else:
            rest = line
        adjectives += [word.strip(" .") for word in rest.split(",") if word.strip()]
    return adjectives[:5]

def generate_prompts(class_name: str, real_desc: str):
    adjectives = get_adjectives(real_desc)
    abnormal_phrases = [f"a {adj} {real_desc}" for adj in adjectives]
    templates = ["{}", "a photo of {}"]
    abnormal_prompts = [template.format(p) for p in abnormal_phrases for template in templates]

    # normal prompt from original AA-CLIP setup
    normal_phrases = [real_desc, f"a {real_desc}", f"the {real_desc}"]
    normal_prompts = [template.format(p) for p in normal_phrases for template in templates]

    os.makedirs("prompts", exist_ok=True)
    with open(f"prompts/{class_name}.json", "w") as f:
        json.dump({"normal": normal_prompts, "abnormal": abnormal_prompts}, f, indent=2)
    print(f"Saved prompts to prompts/{class_name}.json")

if __name__ == "__main__":
    generate_prompts("macaroni", "orange macaroni")
