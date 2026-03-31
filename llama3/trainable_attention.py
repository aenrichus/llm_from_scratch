# Llama 3 Attention Implementation
# This file is a modified copy of ../gpt2/trainable_attention.py
#
# KEY CHANGES FROM GPT-2:
# 1. Standard MHA -> Grouped Query Attention (GQA): K and V use fewer heads than Q
# 2. Learned positional embeddings -> RoPE: position info applied via rotation of Q/K
# 3. No dropout in attention (Llama relies on other regularization)
# 4. No bias in linear projections (Llama uses bias=False everywhere)
# 5. Causal mask computed dynamically instead of stored as buffer

import torch
import torch.nn as nn
from rope import precompute_rope_params, apply_rope


# ============================================================================
# Grouped Query Attention (replaces MultiHeadAttention from GPT-2)
# ============================================================================
# GPT-2 MHA: n_heads Q projections, n_heads K projections, n_heads V projections
# Llama GQA: n_heads Q projections, n_kv_heads K projections, n_kv_heads V projections
#
# The K/V heads are "shared" across groups of Q heads. For example with 32 Q heads
# and 8 KV heads, each KV head serves 4 Q heads. During inference, the KV cache
# stores only 8 heads instead of 32, reducing memory by 4x.
class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.emb_dim = cfg["emb_dim"]
        self.n_heads = cfg["n_heads"]          # Number of query heads (e.g., 32)
        self.n_kv_heads = cfg["n_kv_heads"]    # CHANGED: fewer KV heads (e.g., 8)
        self.head_dim = self.emb_dim // self.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads  # Q heads per KV head group

        # CHANGED: Q projects to full n_heads, but K/V project to fewer n_kv_heads
        # GPT-2: all three project to d_out = n_heads * head_dim
        # Llama: K/V project to n_kv_heads * head_dim (smaller)
        self.W_query = nn.Linear(self.emb_dim, self.n_heads * self.head_dim, bias=False)     # Same as GPT-2
        self.W_key = nn.Linear(self.emb_dim, self.n_kv_heads * self.head_dim, bias=False)    # CHANGED: fewer heads
        self.W_value = nn.Linear(self.emb_dim, self.n_kv_heads * self.head_dim, bias=False)  # CHANGED: fewer heads
        self.out_proj = nn.Linear(self.emb_dim, self.emb_dim, bias=False)  # CHANGED: bias=False (GPT-2 uses bias)

        # REMOVED: self.dropout (Llama doesn't use attention dropout)
        # REMOVED: self.register_buffer("mask", ...) (mask computed dynamically)

    def forward(self, x, cos, sin):  # CHANGED: takes cos, sin for RoPE (GPT-2 takes only x)
        b, num_tokens, _ = x.shape

        # Project to Q, K, V (same concept as GPT-2, different sizes)
        query = self.W_query(x)
        key = self.W_key(x)
        value = self.W_value(x)

        # Reshape to separate heads
        # CHANGED: K/V have n_kv_heads, not n_heads
        query = query.view(b, num_tokens, self.n_heads, self.head_dim).transpose(1, 2)
        key = key.view(b, num_tokens, self.n_kv_heads, self.head_dim).transpose(1, 2)
        value = value.view(b, num_tokens, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # CHANGED: Apply RoPE to Q and K (replaces learned positional embeddings)
        # In GPT-2, positions are added at the input layer via pos_emb.
        # In Llama, positions are encoded by rotating Q/K vectors here.
        query = apply_rope(query, cos, sin)
        key = apply_rope(key, cos, sin)

        # CHANGED: Expand K/V heads to match Q heads via repeat_interleave
        # Each KV head is repeated n_rep times so the shapes align for matmul.
        # This is the core GQA mechanism — storage-efficient but computation-equivalent.
        key = key.repeat_interleave(self.n_rep, dim=1)
        value = value.repeat_interleave(self.n_rep, dim=1)

        # Compute attention scores (same as GPT-2)
        attn_scores = torch.matmul(query, key.transpose(2, 3)) / (self.head_dim ** 0.5)

        # CHANGED: Causal mask computed on the fly (GPT-2 uses a stored buffer)
        mask = torch.triu(torch.ones(num_tokens, num_tokens, device=x.device), diagonal=1).bool()
        attn_scores.masked_fill_(mask, -torch.inf)

        attn_weights = torch.softmax(attn_scores, dim=-1)
        # REMOVED: attn_weights = self.dropout(attn_weights) — Llama has no attention dropout

        # Compute context and project output (same as GPT-2)
        context = torch.matmul(attn_weights, value)
        context = context.transpose(1, 2).contiguous().view(b, num_tokens, self.emb_dim)

        return self.out_proj(context)


# ============================================================================
# Test: Compare GQA with GPT-2's MHA
# ============================================================================
if __name__ == "__main__":
    print("=== Grouped Query Attention (GQA) vs Multi-Head Attention (MHA) ===\n")

    # Test configuration
    cfg = {
        "emb_dim": 64,
        "n_heads": 8,
        "n_kv_heads": 2,  # 4 Q heads share each KV head
        "context_length": 32,
        "rope_base": 500_000,
    }

    # Create GQA module
    torch.manual_seed(123)
    gqa = GroupedQueryAttention(cfg)

    # Create input
    x = torch.randn(2, 10, 64)

    # Precompute RoPE params
    cos, sin = precompute_rope_params(
        head_dim=cfg["emb_dim"] // cfg["n_heads"],
        context_length=cfg["context_length"]
    )

    # Forward pass
    out = gqa(x, cos, sin)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")

    # Count parameters
    gqa_params = sum(p.numel() for p in gqa.parameters())
    print(f"\nGQA parameters (8 Q heads, 2 KV heads): {gqa_params:,}")

    # Compare: if we used full MHA (n_kv_heads == n_heads)
    full_cfg = cfg.copy()
    full_cfg["n_kv_heads"] = full_cfg["n_heads"]
    full_mha = GroupedQueryAttention(full_cfg)
    mha_params = sum(p.numel() for p in full_mha.parameters())
    print(f"Full MHA parameters (8 Q heads, 8 KV heads): {mha_params:,}")
    print(f"Parameter savings from GQA: {(1 - gqa_params / mha_params) * 100:.1f}%")
    print(f"KV cache size reduction: {cfg['n_heads'] / cfg['n_kv_heads']}x")
