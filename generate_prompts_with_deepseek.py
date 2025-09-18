

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
model_id = "deepseek-ai/deepseek-llm-7b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto",
    attn_implementation="sdpa",   # fallback to PyTorch scaled dot product attention
)


# Create generation pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Set item
item = "wood"

# Prompt
prompt = f"""Provide 8 adjectives that describe how defective {item} might look.
Return only a comma-separated list of adjectives.
Do not include explanations, quotes or code.
Adjectives:"""

# Generate
output = pipe(prompt, max_new_tokens=50, do_sample=False, temperature=0.3)[0]["generated_text"]

# Extract adjectives: everything after "Adjectives:"
if "Adjectives:" in output:
    adjectives_text = output.split("Adjectives:")[-1].strip()
else:
    adjectives_text = output.strip()

# Clean and split
adjectives = [adj.strip() for adj in adjectives_text.split(",") if adj.strip()]

# Print only the adjectives
print(adjectives)
