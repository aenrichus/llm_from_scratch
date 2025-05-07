import torch
import torch.nn as nn
from gpt_model import GPTModel

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()

import tiktoken
from gpt_model import generate_text_simple

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # Add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    decoded = tokenizer.decode(flat.tolist())
    return decoded

start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(model=model, 
                                 idx=text_to_token_ids(start_context, tokenizer),
                                 max_new_tokens=10,
                                 context_size=GPT_CONFIG_124M['context_length']
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

# Calculating loss
inputs = torch.tensor([[16833, 3626, 6100],
                       [40, 1107, 588]])
targets = torch.tensor([[3626, 6100, 345],
                         [1107, 588, 11311]])

with torch.no_grad():
    logits = model(inputs)
probabilities = torch.softmax(logits, dim=-1)
print(probabilities.shape)

token_ids = torch.argmax(probabilities, dim=-1, keepdim=True)
print(token_ids)

print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
print(f"Outputs batch 1: {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")

text_idx = 0
target_probs_1 = probabilities[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 1:", target_probs_1)

text_idx = 1
target_probs_2 = probabilities[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 2:", target_probs_2)

log_probs = torch.log(torch.cat([target_probs_1, target_probs_2]))
print("Log probabilities:", log_probs)

avg_log_probs = torch.mean(log_probs)
print("Average log probabilities:", avg_log_probs)

neg_avg_log_probs = avg_log_probs * -1.0
print("Negative average log probabilities:", neg_avg_log_probs)

print("Logits shape:", logits.shape)
print("Targets shape:", targets.shape)

logits_flat = logits.flatten(0, 1)
targets_flat = targets.flatten()
print("Logits flat shape:", logits_flat.shape)
print("Targets flat shape:", targets_flat.shape)

loss = nn.functional.cross_entropy(logits_flat, targets_flat)
print("Loss:", loss)
print(loss.item())