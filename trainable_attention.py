import torch
inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your
     [0.55, 0.87, 0.66], # journey
     [0.57, 0.85, 0.64], # starts
     [0.22, 0.58, 0.33], # with
     [0.77, 0.25, 0.10], # one
     [0.05, 0.80, 0.55]] # step
)

x_2 = inputs[1] # (x)
d_in = inputs.shape[1] # input embedding dimension
d_out = 2 # output embedding dimension

torch.manual_seed(0)
W_query = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)

query_2 = torch.matmul(x_2, W_query)
key_2 = torch.matmul(x_2, W_key)
value_2 = torch.matmul(x_2, W_value)
print("Query:\n", query_2)
print("Key:\n", key_2)
print("Value:\n", value_2)

keys = torch.matmul(inputs, W_key)
values = torch.matmul(inputs, W_value)
print(keys.shape)
print(values.shape)

# Compute attention scores (w)
keys_2 = keys[1]
attn_scores_22 = torch.dot(query_2, keys_2)
print("Attention scores:\n", attn_scores_22) # attention score for query 2 and key 2

attn_scores_2 = torch.matmul(query_2, keys.T)
print("Attention scores:\n", attn_scores_2) # all attention scores for query

# Computer attention weights (a)
d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
print("Attention weights:\n", attn_weights_2) # all attention weights for query

# Compute context vector (z)
context_vector_2 = torch.matmul(attn_weights_2, values)
print("Context vector:\n", context_vector_2) # context vector for query 2

# Compact self-attention class
import torch.nn as nn
class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.randn(d_in, d_out))
        self.W_key = nn.Parameter(torch.randn(d_in, d_out))
        self.W_value = nn.Parameter(torch.randn(d_in, d_out))

    def forward(self, x):
        key = torch.matmul(x, self.W_key)
        query = torch.matmul(x, self.W_query)
        value = torch.matmul(x, self.W_value)

        attn_scores = torch.matmul(query, key.T)
        d_k = key.shape[-1]
        attn_weights = torch.softmax(attn_scores / d_k**0.5, dim=-1)
        context_vector = torch.matmul(attn_weights, value)

        return context_vector

torch.manual_seed(0)
sa_v1 = SelfAttention_v1(d_in, d_out)
print(sa_v1(inputs))
print(context_vector_2)

# Self-attention using Linear layers
class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        key = self.W_key(x)
        query = self.W_query(x)
        value = self.W_value(x)

        attn_scores = torch.matmul(query, key.T)
        d_k = key.shape[-1]
        attn_weights = torch.softmax(attn_scores / d_k**0.5, dim=-1)
        context_vector = torch.matmul(attn_weights, value)

        return context_vector

torch.manual_seed(0)
sa_v2 = SelfAttention_v2(d_in, d_out)
print(sa_v2(inputs))

# Causal attention
queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attn_scores = torch.matmul(queries, keys.T)
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
print(attn_weights)

# Simple context mask
context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
print(mask_simple)

masked_simple = attn_weights * mask_simple
print(masked_simple)

row_sums = torch.sum(masked_simple, dim=-1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
print(masked_simple_norm)

# Context mask using -inf
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_weights.masked_fill(mask.bool(), -torch.inf)
print(masked)

attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=-1)
print(attn_weights)
