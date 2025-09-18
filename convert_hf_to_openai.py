import torch
from collections import OrderedDict

def convert_hf_to_openai(hf_checkpoint_path, output_path):
    print(f"Loading Hugging Face weights from {hf_checkpoint_path}...")
    hf_state_dict = torch.load(hf_checkpoint_path, map_location="cpu")

    # Some HF checkpoints wrap weights inside 'model' or 'state_dict'
    if "state_dict" in hf_state_dict:
        hf_state_dict = hf_state_dict["state_dict"]
    elif "model" in hf_state_dict:
        hf_state_dict = hf_state_dict["model"]

    new_state_dict = OrderedDict()
    print("Converting keys...")

    for k, v in hf_state_dict.items():
        new_k = k

        # Vision model mappings
        new_k = new_k.replace("vision_model.embeddings.patch_embedding.weight", "visual.conv1.weight")
        new_k = new_k.replace("vision_model.embeddings.class_embedding", "visual.class_embedding")
        new_k = new_k.replace("vision_model.embeddings.position_embedding", "visual.positional_embedding")
        new_k = new_k.replace("vision_model.pre_layrnorm", "visual.ln_pre")

        if "encoder.layers" in new_k:
            new_k = new_k.replace("vision_model.encoder.layers.", "visual.transformer.resblocks.")
            new_k = new_k.replace(".layer_norm1", ".ln_1")
            new_k = new_k.replace(".layer_norm2", ".ln_2")
            new_k = new_k.replace(".self_attn.out_proj", ".attn.out_proj")
            new_k = new_k.replace(".self_attn.q_proj", ".attn.q_proj")
            new_k = new_k.replace(".self_attn.k_proj", ".attn.k_proj")
            new_k = new_k.replace(".self_attn.v_proj", ".attn.v_proj")
            new_k = new_k.replace(".mlp.fc1", ".mlp.c_fc")
            new_k = new_k.replace(".mlp.fc2", ".mlp.c_proj")

        # Text model mappings
        new_k = new_k.replace("text_model.embeddings.token_embedding.weight", "token_embedding.weight")
        new_k = new_k.replace("text_model.embeddings.position_embedding", "positional_embedding")
        new_k = new_k.replace("text_model.final_layer_norm", "ln_final")

        if "text_model.encoder.layers" in new_k:
            new_k = new_k.replace("text_model.encoder.layers.", "transformer.resblocks.")
            new_k = new_k.replace(".layer_norm1", ".ln_1")
            new_k = new_k.replace(".layer_norm2", ".ln_2")
            new_k = new_k.replace(".mlp.fc1", ".mlp.c_fc")
            new_k = new_k.replace(".mlp.fc2", ".mlp.c_proj")
            new_k = new_k.replace(".self_attn.q_proj", ".attn.q_proj")
            new_k = new_k.replace(".self_attn.k_proj", ".attn.k_proj")
            new_k = new_k.replace(".self_attn.v_proj", ".attn.v_proj")
            new_k = new_k.replace(".self_attn.out_proj", ".attn.out_proj")

        new_state_dict[new_k] = v

    print(f"Saving converted model to {output_path}...")
    torch.save(new_state_dict, output_path)
    print("Conversion complete.")

if __name__ == "__main__":
    hf_checkpoint_path = "pytorch_model.bin"  
    output_path = "ViT-L-14-336px.pt"
    convert_hf_to_openai(hf_checkpoint_path, output_path)
