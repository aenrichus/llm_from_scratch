# Rotary Position Embeddings (RoPE)
# Unlike GPT-2's learned positional embeddings, RoPE encodes position by rotating
# Q and K vectors in 2D subspaces using sinusoidal frequencies. This means the
# dot product between Q and K depends only on their relative distance, not absolute
# positions — enabling better length generalization and requiring no learned parameters.

import torch


def precompute_rope_params(head_dim, theta_base=500_000, context_length=4096):
    """Precompute the cosine and sine values for RoPE.

    RoPE pairs up dimensions (0,1), (2,3), ..., (d-2, d-1) and rotates each pair
    by a position-dependent angle. Lower-dimensional pairs rotate faster (high frequency),
    higher-dimensional pairs rotate slower (low frequency), creating a multi-scale
    positional encoding.

    Args:
        head_dim: Dimension of each attention head
        theta_base: Base frequency (Llama 3 uses 500,000; original RoPE used 10,000)
        context_length: Maximum sequence length to precompute

    Returns:
        cos, sin: Tensors of shape (context_length, head_dim) for applying rotations
    """
    assert head_dim % 2 == 0, "Head dimension must be even for RoPE"

    # Compute frequency for each pair of dimensions
    # theta_i = 1 / (theta_base ^ (2i / head_dim)) for i = 0, 1, ..., head_dim/2 - 1
    i = torch.arange(0, head_dim, 2).float()  # [0, 2, 4, ..., head_dim-2]
    inv_freq = 1.0 / (theta_base ** (i / head_dim))  # Shape: (head_dim/2,)

    # Compute position * frequency for all positions
    positions = torch.arange(context_length).float()  # [0, 1, 2, ..., context_length-1]
    angles = torch.outer(positions, inv_freq)  # Shape: (context_length, head_dim/2)

    # Duplicate angles for each pair: [theta_0, theta_0, theta_1, theta_1, ...]
    angles = torch.cat([angles, angles], dim=-1)  # Shape: (context_length, head_dim)

    cos = torch.cos(angles)  # Shape: (context_length, head_dim)
    sin = torch.sin(angles)  # Shape: (context_length, head_dim)

    return cos, sin


def apply_rope(x, cos, sin):
    """Apply rotary position embeddings to Q or K tensors.

    For each pair of dimensions (x_0, x_1), the rotation is:
        x_0' = x_0 * cos(theta) - x_1 * sin(theta)
        x_1' = x_0 * sin(theta) + x_1 * cos(theta)

    This is equivalent to 2D rotation matrices applied to each pair of dimensions.

    Args:
        x: Tensor of shape (batch, num_heads, seq_len, head_dim)
        cos: Precomputed cosines of shape (context_length, head_dim)
        sin: Precomputed sines of shape (context_length, head_dim)

    Returns:
        Rotated tensor of same shape as x
    """
    batch, num_heads, seq_len, head_dim = x.shape

    # Select only the positions we need
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)

    # Split into first half and second half of dimensions
    x1 = x[..., :head_dim // 2]  # First half
    x2 = x[..., head_dim // 2:]  # Second half

    # Build the "rotated" version: [-x2, x1] (the perpendicular vector)
    rotated = torch.cat((-x2, x1), dim=-1)

    # Apply rotation: x * cos + rotated * sin
    roped = x * cos + rotated * sin

    return roped.to(x.dtype)


# === Demonstration: RoPE vs Learned Positional Embeddings ===
if __name__ == "__main__":
    # Show how RoPE encodes relative position
    head_dim = 16
    context_length = 32
    cos, sin = precompute_rope_params(head_dim, context_length=context_length)

    print(f"Precomputed cos shape: {cos.shape}")  # (32, 16)
    print(f"Precomputed sin shape: {sin.shape}")  # (32, 16)

    # Simulate Q and K at different positions
    torch.manual_seed(42)
    batch_size = 1
    num_heads = 1
    seq_len = 8
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)

    # Apply RoPE
    q_roped = apply_rope(q, cos, sin)
    k_roped = apply_rope(k, cos, sin)

    print(f"\nOriginal Q shape: {q.shape}")
    print(f"RoPE'd Q shape:   {q_roped.shape}")

    # Key insight: attention scores now encode relative position
    attn_scores = torch.matmul(q_roped, k_roped.transpose(-2, -1))
    print(f"\nAttention scores shape: {attn_scores.shape}")
    print(f"Attention scores (position-aware):\n{attn_scores[0, 0]}")
