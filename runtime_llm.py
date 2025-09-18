import os
import re
import threading
import traceback
from functools import lru_cache
from typing import List, Optional
from importlib.util import find_spec

LLM_DEVICE = os.environ.get("AA_LLM_DEVICE", "cpu")  # default to CPU

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Toggle with env var: AA_USE_LLM=1 enables LLM; default off.
USE_LLM = os.environ.get("AA_USE_LLM", "0") == "1"
MODEL_ID = os.environ.get("AA_LLM_MODEL", "deepseek-ai/deepseek-llm-7b-chat")

_PIPE = None
_LOCK = threading.Lock()


# === Class-specific prompt templates ===
# {item} should be the REAL_NAMES description 
DEFAULT_PROMPT = (
        "Give me 7 adjectives that describe how defective {item} might look.\n"
        "Return only a comma-separated list of adjectives (e.g., cracked, melted, dirty).\n"
        "Do not include explanations, quotes, or code.\n"
        "Adjectives:"
        )

CLASS_PROMPTS = {
    "MVTec": {
        "bottle": """Give 8 adjectives that describe how a damaged or defective {item} might look (appearance only).
        Return only a comma-separated list of adjectives.
        Do not include explanations, quotes or code.
        Adjectives:""",        
        
        "cable": """List 8 adjectives describing material or coating defects in {item} (plastic or rubber insulation).
        Return only a comma-separated list of adjectives.
        Do not include explanations, quotes or code.
        Adjectives:""",        

        "capsule": """Give me 8 visual defects that can occur in a medicine {item}.
        Return only a comma-separated list of adjectives.
        Do not include explanations, quotes or code.
        Adjectives:""", 

        "carpet": """Give me 8 visual defects that can occur in a {item}.
        Return only a comma-separated list of adjectives(e.g., torn, dirty).
        Do not include explanations, quotes or code.
        Adjectives:""",                 

        "grid": """List 8 adjectives describing common geometric defects in a {item}.
        Return only a comma-separated list of adjectives.
        Do not include explanations, quotes or code.
        Adjectives:""", 

        "hazelnut": """Give me 8 adjectives describing how a defective {item} might look,focusing on surface texture, color, and shape.
        Return only a comma-separated list of adjectives.
        Do not include explanations, quotes or code.
        Adjectives:""", 

        "leather": """Give me 8 adjectives describing how defective {item} might look, focusing on surface texture, cracks, stains, and wear.
        Return only a comma-separated list of adjectives.
        Do not include explanations, quotes or code.
        Adjectives:""",  

        "metal_nut": """Give me 8 visual defects that can occur in a {item}.
        Return only a comma-separated list of adjectives (e.g. scratched, dented).
        Do not include explanations, quotes or code.
        Adjectives:""", 

        "pill": """Give me 8 adjectives that describe how a defective pharmaceutical {item} might look, focusing on surface texture, color changes, and shape irregularities.
        Return only a comma-separated list of adjectives.
        Do not include explanations, quotes or code.
        Adjectives:""",  

        "screw": """Give me 8 visual defects that can occur in a {item}.
        Return only a comma-separated list of adjectives.
        Do not include explanations, quotes or code.
        Adjectives:""", 

        "tile": """Give me 8 visual defects that can occur in a {item}.
        Return only a comma-separated list of adjectives.
        Do not include explanations, quotes or code.
        Adjectives:""", 

        "transistor": """Give me 8 adjectives that describe visible surface defects in a defective {item}, focus only on appearance.
        Return only a comma-separated list of adjectives.
        Do not include explanations, quotes or code.
        Adjectives:""", 

        "toothbrush": """List 8 unique adjectives that describe visual defects in a {item} toothbrush.
        Focus on bristle wear (e.g., bent, frayed), missing or broken parts, and surface discoloration.
        Return only a comma-separated list of adjectives.
        Do not include explanations, quotes or code.
        Adjectives:""", 

        "wood": """List 8 adjectives describing visible defects in a wooden surface (appearance only).
        Return only a comma-separated list of adjectives.
        Do not include explanations, quotes or code.
        Adjectives:""", 

        "zipper": """Give me 8 visual defects that can occur in a {item}.
        Return only a comma-separated list of adjectives.
        Do not include explanations, quotes or code.
        Adjectives:""", 
    }
}




def _get_pipe():
    global _PIPE
    if _PIPE is None:
        with _LOCK:
            if _PIPE is None:
                tok = AutoTokenizer.from_pretrained(MODEL_ID)
                kwargs = {"torch_dtype": "auto"}
                # If you *really* want GPU, set AA_LLM_DEVICE=cuda and keep accelerate
                if LLM_DEVICE == "cuda" and find_spec("accelerate") is not None:
                    kwargs["device_map"] = "auto"
                mdl = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
                # critical: keep pipeline on CPU unless explicitly asked for GPU
                pipe_device = -1 if LLM_DEVICE != "cuda" else 0
                _PIPE = pipeline("text-generation", model=mdl, tokenizer=tok, device=pipe_device)
    return _PIPE



@lru_cache(maxsize=512)
def get_visual_defect_adjectives(
    item: str,
    class_name: Optional[str] = None,
    dataset: str = "MVTec",
) -> List[str]:
    """
    Return up to 8 adjectives describing VISUAL defects for `item`.
    `item` should be the REAL_NAMES description (e.g., 'gray carpet').
    If LLM is disabled or anything goes wrong, return [].
    """
    if not USE_LLM:
        print("[LLM INFO] AA_USE_LLM != 1, returning [].")
        return []

    try:
        pipe = _get_pipe()

        # choose a class-specific prompt if available, else fallback
        tmpl = DEFAULT_PROMPT
        if class_name is not None:
            tmpl = CLASS_PROMPTS.get(dataset, {}).get(class_name, DEFAULT_PROMPT)

        prompt = tmpl.format(item=item)

        out = pipe(prompt, max_new_tokens=64, do_sample=False, temperature=0.3)[0]["generated_text"]
        completion = out[len(prompt):].strip()
     
        
        text = completion.split("Adjectives:", 1)[-1].strip() if "Adjectives:" in completion else completion
        # split, clean, de-duplicate, keep order
        raw = [w.strip() for w in text.split(",") if w.strip()]
        seen = set()
        cleaned = []
        for w in raw:
            token = re.sub(r"[^A-Za-z\- ]", "", w).strip().lower()
            if not token:
                continue
            # drop non-visual 
            if any(bad in token for bad in ("adjectives", "quotes", "code", "eg", "return only")):
                continue
            if token in seen:
                continue
            seen.add(token)
            cleaned.append(token)
            if len(cleaned) == 8:
                break

        if len(cleaned) < 3:
            cleaned = []
        return cleaned        
    except Exception as e:
        print(f"[LLM ERROR] generation failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        return []


   

