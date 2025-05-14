# instruction.py (or wherever you want the biopsy)
import torch
from gpt_model import GPTModel  # adjust import path if needed

# 1. Instantiate your GPTModel with the same config you used for loading weights
GPT_CONFIG_M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 1024,      # GPT-2 medium hidden size
    "n_heads": 16,        # make sure this matches your model
    "n_layers": 24,       # "
    "drop_rate": 0.1,
    "qkv_bias": True
    }

model = GPTModel(GPT_CONFIG_M)

# 2. Load your pretrained medium weights
state = torch.load("gpt2-medium355M_sft.pth", map_location="cpu")
model.load_state_dict(state)
model.eval()

# 3) Embedding shapes
print(" tok_emb weight shape :", model.tok_emb.weight.shape)
print(" pos_emb weight shape :", model.pos_emb.weight.shape)

# 4) Run a tiny forward through just the embedding layer:
dummy_ids = torch.tensor([[0, 1, 2]])
with torch.no_grad():
    h = model.tok_emb(dummy_ids) + model.pos_emb(torch.arange(3)[None, :])
print(" embed output shape   :", h.shape)   # should be [1, 3, 1024]

# 1) Print out the FeedForward submodule so we see its API:
ffn = model.trf_blocks[0].ffn
print("FFN structure:\n", ffn)

# 2) List all parameter names and shapes:
for name, param in ffn.named_parameters():
    print(f"{name:15s} → {tuple(param.shape)}")


# 4. First block’s MLP c_fc
#    adjust if your block API is different
first_block = model.trf_blocks[0]
print(" first block c_fc    :", first_block.ffn.c_fc.weight.shape)  
#                                        ^^^^^          ^^^^^
#                                         block list     mlp layer

# If any of these print 256, you know it’s being sliced or projected
# somewhere in your forward path. 
