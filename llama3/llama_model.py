# Llama 3.2 Model Implementation
# This file is a modified copy of ../gpt_model.py
#
# Every change from the GPT-2 original is marked with "CHANGED:", "ADDED:", or "REMOVED:"
# comments so you can see exactly what's different and why.
#
# Summary of changes:
#   1. LayerNorm     -> RMSNorm       (simpler, faster normalization)
#   2. Learned PosEmb -> RoPE         (relative position encoding via rotation)
#   3. GELU FFN      -> SwiGLU FFN    (gated activation for better expressiveness)
#   4. Standard MHA  -> GQA           (grouped query attention for efficient KV cache)
#   5. No dropout    (Llama uses other regularization)
#   6. No bias       (all nn.Linear use bias=False)

import torch
import torch.nn as nn

# CHANGED: Llama 3.2 1B configuration (compare with GPT_CONFIG_124M in gpt_model.py)
# GPT-2 124M: vocab=50257, ctx=1024, emb=768, heads=12, layers=12
LLAMA32_CONFIG = {
    "vocab_size": 128_256,     # CHANGED: Larger vocabulary (GPT-2: 50,257)
    "context_length": 8192,    # CHANGED: 8x longer context (GPT-2: 1,024)
    "emb_dim": 2048,           # CHANGED: Larger embedding dim (GPT-2: 768)
    "n_heads": 32,             # CHANGED: More Q heads (GPT-2: 12)
    "n_kv_heads": 8,           # ADDED: GQA — fewer KV heads (GPT-2: n_kv_heads == n_heads)
    "n_layers": 16,            # CHANGED: More layers (GPT-2: 12)
    "hidden_dim": 8192,        # CHANGED: FFN hidden dim, ~4x emb_dim (GPT-2: 4 * emb_dim = 3072)
    "dtype": torch.bfloat16,   # ADDED: Llama uses bfloat16 for efficiency
    "rope_base": 500_000,      # ADDED: RoPE base frequency
    "rope_freq": None,         # ADDED: Optional RoPE frequency scaling
}

# REMOVED: DummyGPTModel, DummyTransformerBlock, DummyLayerNorm (not needed)


# ============================================================================
# CHANGED: RMSNorm replaces LayerNorm
# ============================================================================
# GPT-2 LayerNorm (from gpt_model.py lines 97-108):
#   mean = x.mean(dim=-1, keepdim=True)
#   var = x.var(dim=-1, keepdim=True)
#   norm_x = (x - mean) / sqrt(var + eps)
#   return scale * norm_x + shift          <-- has both scale AND shift
#
# Llama RMSNorm:
#   rms = sqrt(mean(x^2) + eps)
#   norm_x = x / rms
#   return scale * norm_x                  <-- scale only, NO shift
#
# Why? Mean subtraction contributes almost nothing to training stability.
# Removing it saves ~15% compute with no quality loss.
class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        # REMOVED: self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        # CHANGED: No mean subtraction, use RMS instead of variance
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        norm_x = x / rms
        return self.scale * norm_x
        # REMOVED: + self.shift


# ============================================================================
# REMOVED: GELU activation class (GPT-2's gpt_model.py lines 120-125)
# Llama uses SiLU (Swish) instead, which is built into PyTorch as F.silu()
# GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
# SiLU(x) = x * sigmoid(x)  <-- simpler, used in the gating mechanism
# ============================================================================


# ============================================================================
# CHANGED: SwiGLU FeedForward replaces GELU FeedForward
# ============================================================================
# GPT-2 FeedForward (from gpt_model.py lines 147-157):
#   Linear(emb_dim -> 4*emb_dim) -> GELU -> Linear(4*emb_dim -> emb_dim)
#   Two matrices, 4x expansion, GELU activation
#
# Llama SwiGLU FeedForward:
#   gate = SiLU(Linear_gate(x)) * Linear_up(x)
#   output = Linear_down(gate)
#   Three matrices, ~2.67x expansion, SiLU+gating mechanism
#
# Why? The gating (element-wise multiply of two paths) lets the network
# learn to selectively pass information, giving more expressive power.
# Models train to the same loss faster with SwiGLU.
class SwiGLUFeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # CHANGED: Three linear layers instead of two
        # CHANGED: bias=False everywhere (GPT-2 uses bias=True)
        # CHANGED: expansion to hidden_dim (configurable) instead of 4*emb_dim
        self.gate_proj = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], bias=False)  # ADDED: gate path
        self.up_proj = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], bias=False)    # Like GPT-2's first linear
        self.down_proj = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], bias=False)  # Like GPT-2's second linear

    def forward(self, x):
        # CHANGED: SiLU gating instead of simple GELU
        # GPT-2: return Linear2(GELU(Linear1(x)))
        # Llama: return down(SiLU(gate(x)) * up(x))
        gate = torch.nn.functional.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


# ============================================================================
# CHANGED: Transformer block uses new components
# ============================================================================
# Structure is the same pre-norm pattern as GPT-2, but with different components:
# GPT-2:  LayerNorm -> MHA      -> Dropout -> Residual -> LayerNorm -> GELU FFN    -> Dropout -> Residual
# Llama:  RMSNorm   -> GQA+RoPE ->           Residual -> RMSNorm   -> SwiGLU FFN ->           Residual
from trainable_attention import GroupedQueryAttention  # CHANGED: import GQA instead of MHA

class LlamaTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = GroupedQueryAttention(cfg)    # CHANGED: GQA replaces MultiHeadAttention
        self.ffn = SwiGLUFeedForward(cfg)       # CHANGED: SwiGLU replaces FeedForward
        self.ln1 = RMSNorm(cfg["emb_dim"])      # CHANGED: RMSNorm replaces LayerNorm
        self.ln2 = RMSNorm(cfg["emb_dim"])      # CHANGED: RMSNorm replaces LayerNorm
        # REMOVED: self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, cos, sin):  # CHANGED: takes cos, sin for RoPE
        # Pre-norm attention with residual (same pattern as GPT-2)
        shortcut = x
        x = self.ln1(x)
        x = self.att(x, cos, sin)  # CHANGED: pass RoPE params to attention
        # REMOVED: x = self.drop_shortcut(x)
        x = x + shortcut

        # Pre-norm FFN with residual (same pattern as GPT-2)
        shortcut = x
        x = self.ln2(x)
        x = self.ffn(x)
        # REMOVED: x = self.drop_shortcut(x)
        x = x + shortcut

        return x


# ============================================================================
# CHANGED: Full Llama Model (replaces GPTModel)
# ============================================================================
# Key differences from GPTModel (gpt_model.py lines 253-274):
# 1. No pos_emb — positional info comes from RoPE inside attention
# 2. No dropout on embeddings
# 3. Uses nn.ModuleList instead of nn.Sequential (blocks need extra args)
# 4. RoPE cos/sin precomputed and stored as buffers
from rope import precompute_rope_params

class LlamaModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Token embedding (same as GPT-2)
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        # REMOVED: self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        # REMOVED: self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # CHANGED: ModuleList instead of Sequential (blocks need cos/sin args)
        self.trf_blocks = nn.ModuleList(
            [LlamaTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = RMSNorm(cfg["emb_dim"])  # CHANGED: RMSNorm replaces LayerNorm
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

        # ADDED: Precompute RoPE frequencies (replaces learned positional embeddings)
        head_dim = cfg["emb_dim"] // cfg["n_heads"]
        cos, sin = precompute_rope_params(
            head_dim=head_dim,
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"]
        )
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape

        # CHANGED: Only token embeddings, no positional embeddings added here
        # GPT-2: x = tok_emb(in_idx) + pos_emb(torch.arange(seq_len))
        # Llama: x = tok_emb(in_idx)  — position info comes from RoPE in attention
        x = self.tok_emb(in_idx)
        # REMOVED: pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        # REMOVED: x = tok_embeds + pos_embeds
        # REMOVED: x = self.drop_emb(x)

        # CHANGED: Explicit loop (not Sequential) because blocks need cos/sin
        for block in self.trf_blocks:
            x = block(x, self.cos, self.sin)

        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


# ============================================================================
# Simple method of generating text (copied from gpt_model.py, adapted for Llama)
# ============================================================================
def generate_text_simple(model, idx, max_new_tokens, context_size):
    # Same logic as GPT-2's generate_text_simple in gpt_model.py
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probs, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


# ============================================================================
# Tests: Verify each component and compare with GPT-2
# ============================================================================
if __name__ == "__main__":
    # === Test RMSNorm vs LayerNorm ===
    print("=" * 60)
    print("TEST 1: RMSNorm (Llama) vs LayerNorm (GPT-2)")
    print("=" * 60)
    torch.manual_seed(123)
    batch_example = torch.randn(2, 5)

    rms_norm = RMSNorm(emb_dim=5)
    out_rms = rms_norm(batch_example)
    print(f"Input:       {batch_example[0].tolist()}")
    print(f"RMSNorm out: {out_rms[0].tolist()}")
    rms_check = torch.sqrt(torch.mean(out_rms ** 2, dim=-1))
    print(f"Output RMS per sample: {rms_check.tolist()} (should be ~1.0)")
    print()

    # === Test SwiGLU FeedForward ===
    print("=" * 60)
    print("TEST 2: SwiGLU FeedForward (Llama) vs GELU FFN (GPT-2)")
    print("=" * 60)
    test_cfg = {"emb_dim": 64, "hidden_dim": 172}
    swiglu = SwiGLUFeedForward(test_cfg)
    x = torch.randn(2, 3, 64)
    out = swiglu(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    swiglu_params = sum(p.numel() for p in swiglu.parameters())
    print(f"SwiGLU params: {swiglu_params:,} (3 matrices: gate + up + down)")
    print(f"GPT-2 FFN would have: {64*256 + 256 + 256*64 + 64:,} params (2 matrices, 4x expansion)")
    print()

    # === Test GQA ===
    print("=" * 60)
    print("TEST 3: Grouped Query Attention (Llama) vs MHA (GPT-2)")
    print("=" * 60)
    gqa_cfg = {
        "emb_dim": 64,
        "n_heads": 8,
        "n_kv_heads": 2,
        "context_length": 32,
        "rope_base": 500_000,
    }
    from rope import precompute_rope_params
    cos, sin = precompute_rope_params(head_dim=64 // 8, context_length=32)
    gqa = GroupedQueryAttention(gqa_cfg)
    x = torch.randn(2, 10, 64)
    out = gqa(x, cos, sin)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Q heads: {gqa_cfg['n_heads']}, KV heads: {gqa_cfg['n_kv_heads']}")
    print(f"KV cache reduction: {gqa_cfg['n_heads'] // gqa_cfg['n_kv_heads']}x")
    print()

    # === Test Full Model ===
    print("=" * 60)
    print("TEST 4: Full Llama Model")
    print("=" * 60)
    small_cfg = {
        "vocab_size": 1000,
        "context_length": 64,
        "emb_dim": 64,
        "n_heads": 8,
        "n_kv_heads": 2,
        "n_layers": 4,
        "hidden_dim": 172,
        "dtype": torch.float32,
        "rope_base": 500_000,
        "rope_freq": None,
    }

    torch.manual_seed(123)
    model = LlamaModel(small_cfg)
    input_ids = torch.randint(0, 1000, (2, 16))
    logits = model(input_ids)
    print(f"Input shape:  {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print()

    # === Test Text Generation ===
    print("=" * 60)
    print("TEST 5: Text Generation (random weights)")
    print("=" * 60)
    model.eval()
    start_ids = torch.randint(0, 1000, (1, 3))
    generated = generate_text_simple(model, start_ids, max_new_tokens=10, context_size=small_cfg["context_length"])
    print(f"Start tokens:     {start_ids[0].tolist()}")
    print(f"Generated tokens: {generated[0].tolist()}")
    print(f"(Random outputs expected — no training or pretrained weights loaded)")
    print()

    # === Show Configuration Comparison ===
    print("=" * 60)
    print("CONFIGURATION COMPARISON: GPT-2 124M vs Llama 3.2 1B")
    print("=" * 60)
    gpt2_config = {
        "vocab_size": 50_257,
        "context_length": 1_024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_kv_heads": 12,
        "n_layers": 12,
        "ffn_dim": "4 * 768 = 3,072",
        "normalization": "LayerNorm",
        "activation": "GELU",
        "position_encoding": "Learned",
        "attention": "Full MHA",
    }
    llama_config_display = {
        "vocab_size": "128,256",
        "context_length": "8,192",
        "emb_dim": "2,048",
        "n_heads": 32,
        "n_kv_heads": 8,
        "n_layers": 16,
        "ffn_dim": "8,192",
        "normalization": "RMSNorm",
        "activation": "SiLU (SwiGLU)",
        "position_encoding": "RoPE",
        "attention": "GQA",
    }
    print(f"{'Component':<22} {'GPT-2 124M':<22} {'Llama 3.2 1B':<22}")
    print("-" * 66)
    for key in gpt2_config:
        print(f"{key:<22} {str(gpt2_config[key]):<22} {str(llama_config_display[key]):<22}")
