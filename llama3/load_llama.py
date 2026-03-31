# Loading pretrained Llama 3.2 weights from Hugging Face
# This is the Llama equivalent of ../gpt2/load_pretrained.py
#
# GPT-2 loads weights from OpenAI's TensorFlow checkpoints.
# Llama loads weights from Hugging Face's safetensors format.
#
# Prerequisites:
#   pip install transformers safetensors sentencepiece
#   You need access to meta-llama/Llama-3.2-1B on Hugging Face
#   (requires accepting Meta's license agreement)

import torch
import torch.nn as nn
from llama_model import LlamaModel, LLAMA32_CONFIG, generate_text_simple


def assign(left, right):
    """Assign pretrained weights to model parameters (same pattern as GPT-2 loader)."""
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch: {left.shape} vs {right.shape}")
    return nn.Parameter(right.clone().detach())


def load_weights_from_hf(model, model_name="meta-llama/Llama-3.2-1B"):
    """Load pretrained Llama weights from a Hugging Face checkpoint.

    This function maps HuggingFace's weight names to our model's weight names.
    The mapping is straightforward because our model mirrors the official architecture.

    HuggingFace naming convention:
        model.embed_tokens.weight           -> tok_emb.weight
        model.layers.{i}.self_attn.q_proj   -> trf_blocks[i].att.W_query
        model.layers.{i}.self_attn.k_proj   -> trf_blocks[i].att.W_key
        model.layers.{i}.self_attn.v_proj   -> trf_blocks[i].att.W_value
        model.layers.{i}.self_attn.o_proj   -> trf_blocks[i].att.out_proj
        model.layers.{i}.mlp.gate_proj      -> trf_blocks[i].ffn.gate_proj
        model.layers.{i}.mlp.up_proj        -> trf_blocks[i].ffn.up_proj
        model.layers.{i}.mlp.down_proj      -> trf_blocks[i].ffn.down_proj
        model.layers.{i}.input_layernorm    -> trf_blocks[i].ln1
        model.layers.{i}.post_attention_layernorm -> trf_blocks[i].ln2
        model.norm.weight                   -> final_norm.scale
        lm_head.weight                      -> out_head.weight
    """
    try:
        from safetensors.torch import load_file
        from pathlib import Path
        import json
    except ImportError:
        raise ImportError(
            "Please install safetensors: pip install safetensors"
        )

    # Try loading from a local directory or download from HF
    try:
        from huggingface_hub import snapshot_download
        model_path = snapshot_download(repo_id=model_name)
    except ImportError:
        # If huggingface_hub not installed, assume model_name is a local path
        model_path = model_name

    # Load all safetensor files
    from pathlib import Path
    model_path = Path(model_path)
    safetensor_files = list(model_path.glob("*.safetensors"))

    if not safetensor_files:
        raise FileNotFoundError(f"No safetensors files found in {model_path}")

    # Merge all shards into one state dict
    combined_weights = {}
    for sf_file in safetensor_files:
        combined_weights.update(load_file(sf_file))

    # Load token embeddings
    model.tok_emb.weight = assign(
        model.tok_emb.weight,
        combined_weights["model.embed_tokens.weight"]
    )

    # Load transformer blocks
    for i in range(len(model.trf_blocks)):
        # Attention weights
        model.trf_blocks[i].att.W_query.weight = assign(
            model.trf_blocks[i].att.W_query.weight,
            combined_weights[f"model.layers.{i}.self_attn.q_proj.weight"]
        )
        model.trf_blocks[i].att.W_key.weight = assign(
            model.trf_blocks[i].att.W_key.weight,
            combined_weights[f"model.layers.{i}.self_attn.k_proj.weight"]
        )
        model.trf_blocks[i].att.W_value.weight = assign(
            model.trf_blocks[i].att.W_value.weight,
            combined_weights[f"model.layers.{i}.self_attn.v_proj.weight"]
        )
        model.trf_blocks[i].att.out_proj.weight = assign(
            model.trf_blocks[i].att.out_proj.weight,
            combined_weights[f"model.layers.{i}.self_attn.o_proj.weight"]
        )

        # FFN weights (SwiGLU: gate, up, down projections)
        model.trf_blocks[i].ffn.gate_proj.weight = assign(
            model.trf_blocks[i].ffn.gate_proj.weight,
            combined_weights[f"model.layers.{i}.mlp.gate_proj.weight"]
        )
        model.trf_blocks[i].ffn.up_proj.weight = assign(
            model.trf_blocks[i].ffn.up_proj.weight,
            combined_weights[f"model.layers.{i}.mlp.up_proj.weight"]
        )
        model.trf_blocks[i].ffn.down_proj.weight = assign(
            model.trf_blocks[i].ffn.down_proj.weight,
            combined_weights[f"model.layers.{i}.mlp.down_proj.weight"]
        )

        # RMSNorm weights (scale only, no shift — unlike GPT-2's LayerNorm)
        model.trf_blocks[i].ln1.scale = assign(
            model.trf_blocks[i].ln1.scale,
            combined_weights[f"model.layers.{i}.input_layernorm.weight"]
        )
        model.trf_blocks[i].ln2.scale = assign(
            model.trf_blocks[i].ln2.scale,
            combined_weights[f"model.layers.{i}.post_attention_layernorm.weight"]
        )

    # Final layer norm
    model.final_norm.scale = assign(
        model.final_norm.scale,
        combined_weights["model.norm.weight"]
    )

    # Output head
    model.out_head.weight = assign(
        model.out_head.weight,
        combined_weights["lm_head.weight"]
    )

    return model


# ============================================================================
# Usage example (requires downloading Llama 3.2 1B weights from HuggingFace)
# ============================================================================
if __name__ == "__main__":
    import sys

    # Check if model path was provided
    if len(sys.argv) < 2:
        print("Usage: python load_llama.py <path_to_llama_weights>")
        print()
        print("To get weights:")
        print("  1. Accept Meta's license at https://huggingface.co/meta-llama/Llama-3.2-1B")
        print("  2. pip install huggingface_hub safetensors")
        print("  3. python load_llama.py meta-llama/Llama-3.2-1B")
        print()
        print("Running architecture test with random weights instead...")
        print()

        # Test with random weights to verify the architecture works
        model = LlamaModel(LLAMA32_CONFIG)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Llama 3.2 1B architecture created successfully!")
        print(f"Total parameters: {total_params / 1e9:.2f}B")
        print(f"Model size (float32): {total_params * 4 / (1024**3):.2f} GB")
        print(f"Model size (bfloat16): {total_params * 2 / (1024**3):.2f} GB")
        sys.exit(0)

    model_path = sys.argv[1]

    # Create model with Llama 3.2 1B config
    print("Creating Llama 3.2 1B model...")
    model = LlamaModel(LLAMA32_CONFIG)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e9:.2f}B")

    # Load pretrained weights
    print(f"Loading weights from {model_path}...")
    load_weights_from_hf(model, model_path)
    print("Weights loaded successfully!")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Tokenize and generate
    # Llama 3 uses a SentencePiece-based tokenizer (not tiktoken like GPT-2)
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        prompt = "The meaning of life is"
        print(f"\nPrompt: {prompt}")
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Generate text
        torch.manual_seed(123)
        output_ids = generate_text_simple(
            model=model,
            idx=input_ids,
            max_new_tokens=50,
            context_size=LLAMA32_CONFIG["context_length"]
        )
        generated_text = tokenizer.decode(output_ids[0].tolist())
        print(f"Generated: {generated_text}")

    except ImportError:
        print("\nInstall transformers for tokenizer: pip install transformers")
        print("Testing with random token IDs instead...")
        input_ids = torch.randint(0, LLAMA32_CONFIG["vocab_size"], (1, 5)).to(device)
        output_ids = generate_text_simple(model, input_ids, max_new_tokens=10, context_size=LLAMA32_CONFIG["context_length"])
        print(f"Generated token IDs: {output_ids[0].tolist()}")
