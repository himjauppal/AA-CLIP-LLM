# text_refiner.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load GPT-mini
tok = AutoTokenizer.from_pretrained("erwanf/gpt2-mini")
gptm = AutoModelForCausalLM.from_pretrained("erwanf/gpt2-mini").to(device).eval()

@torch.inference_mode()
def improve_adj(phrase: str) -> str:
    """
    Rewrites the defect adjective for better wording using GPT-mini.
    Example: 'broken carpet' -> 'torn carpet'
    """
    prompt = f"Replace the adjective in '{phrase}' with a better adjective that describes the noun accurately. Return only the new phrase."
    inputs = tok(prompt, return_tensors="pt").to(device)
    outputs = gptm.generate(**inputs, max_new_tokens=6, temperature=0.7,
                             do_sample=True, eos_token_id=tok.eos_token_id)
    refined = tok.decode(outputs[0], skip_special_tokens=True).strip()
    return refined.lower()


if "__name__" == "__main__":
    improved_results = improve_adj(phrase = "This is a broken carpet")
    print("Improved")