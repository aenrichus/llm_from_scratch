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

file_path = 'decay.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    raw_text = file.read()

total_characters = len(raw_text)
total_tokens = len(tokenizer.encode(raw_text))
print(f"Total characters: {total_characters}")
print(f"Total tokens: {total_tokens}")

train_ratio = 0.9
split_idx = int(total_characters * train_ratio)
train_data = raw_text[:split_idx]
val_data = raw_text[split_idx:]
print(f"Training data size: {len(train_data)}")
print(f"Validation data size: {len(val_data)}")

from dataloader import create_dataloader_v1
torch.manual_seed(123)

train_loader = create_dataloader_v1(
    train_data, 
    batch_size=2, 
    max_length=GPT_CONFIG_124M["context_length"], 
    stride=GPT_CONFIG_124M["context_length"], 
    shuffle=True, 
    drop_last=True, 
    num_workers=0
)
val_loader = create_dataloader_v1(
    val_data, 
    batch_size=2, 
    max_length=GPT_CONFIG_124M["context_length"], 
    stride=GPT_CONFIG_124M["context_length"], 
    shuffle=False, 
    drop_last=False, 
    num_workers=0
)

print("Train loader:")
for x, y in train_loader:
    print(x.shape, y.shape)

print("Validation loader:")
for x, y in val_loader:
    print(x.shape, y.shape)

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    logits_flat = logits.flatten(0, 1)
    targets_flat = target_batch.flatten()
    loss = nn.functional.cross_entropy(logits_flat, targets_flat)
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0

    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    
    avg_loss = total_loss / num_batches
    return avg_loss

device = torch.device("mps" if torch.mps.is_available() else "cpu")
print("Device:", device)

model.to(device)
with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)
print(f"Train loss: {train_loss:.4f}")
print(f"Validation loss: {val_loss:.4f}")

# Training an LLM
def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()

            # Logging gradients for debugging
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad = param.grad
                    print(f"{name}: grad mean={grad.mean():.6f}, std={grad.std():.6f}, max={grad.abs().max():.6f}")
                    grad_norm = grad.norm().item()
                    print(f"{name}: grad norm = {grad_norm:.4f}")
                    if grad_norm > 1000:  # Adjust threshold as needed
                        print(f"🚨 Warning: Exploding gradient in {name}!")
                else:
                    print(f"{name}: grad is None")

            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Epoch* {epoch+1}, (Step {global_step:06d}) :" 
                      f"Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}")
        
        generate_and_print_sample(model, tokenizer, device, start_context)
    return train_losses, val_losses, track_tokens_seen

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(model=model,
                                         idx=encoded,
                                         max_new_tokens=50,
                                         context_size=context_size)
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-1)
num_epochs = 0
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device, num_epochs=num_epochs,
    eval_freq=5, eval_iter=5, start_context="Every effort moves you", tokenizer=tokenizer)

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label='Train Loss', color='blue')
    ax1.plot(epochs_seen, val_losses, label='Validation Loss', color='orange')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper right')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel('Tokens Seen')
    fig.tight_layout()
    # plt.show() # Uncomment to show the plot

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

model.to("cpu")
model.eval()
tokenizer = tiktoken.get_encoding("gpt2")
token_ids = generate_text_simple(
    model=model, 
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=25,
    context_size=GPT_CONFIG_124M['context_length']
)
print(token_ids_to_text(token_ids, tokenizer))

## Temperature scaling
vocab = {
    "closer": 0,
    "every": 1,
    "effort": 2,
    "forward": 3,
    "inches": 4,
    "moves": 5,
    "pizza": 6,
    "toward": 7,
    "you": 8,
}
inverse_vocab = {v: k for k, v in vocab.items()}

next_token_logits = torch.tensor([4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79])

probabilities = torch.softmax(next_token_logits, dim=0)
next_token_id = torch.argmax(probabilities).item()
print(inverse_vocab[next_token_id])

torch.manual_seed(123)
next_token_id = torch.multinomial(probabilities, num_samples=1).item()
print(inverse_vocab[next_token_id])

def print_sampled_tokens(probabilities):
    torch.manual_seed(123)
    sample = [torch.multinomial(probabilities, num_samples=1).item() for i in range(1_000)]
    sampled_ids = torch.bincount(torch.tensor(sample))
    for i, freq in enumerate(sampled_ids):
        if freq > 0:
            print(f"{inverse_vocab[i]}: {freq.item()}")

print_sampled_tokens(probabilities)

def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    probabilities = torch.softmax(scaled_logits, dim=0)
    return probabilities

temperatures = [1, 0.1, 5]
scaled_probabilities = [softmax_with_temperature(next_token_logits, T) for T in temperatures]
x = torch.arange(len(vocab))
bar_width = 0.15
fig, ax = plt.subplots(figsize=(5, 3))
for i, T in enumerate(temperatures):
    rects = ax.bar(x + i * bar_width, scaled_probabilities[i], width=bar_width, label=f'Temperature={T}')
ax.set_ylabel('Probability')
ax.set_xticks(x)
ax.set_xticklabels(vocab.keys(), rotation=90)
ax.legend()
plt.tight_layout()
# plt.show() # Uncomment to show the plot

## Top-k sampling
top_k = 3
top_logits, top_pos = torch.topk(next_token_logits, top_k)
print("Top logits:", top_logits)
print("Top positions:", top_pos)

new_logits = torch.where(
    condition=next_token_logits < top_logits[-1],
    input=torch.tensor(float("-inf")),
    other=next_token_logits
)
print("New logits:", new_logits)

topk_probabilities = torch.softmax(new_logits, dim=0)
print("Top-k probabilities:", topk_probabilities)

# Modify text generation function to include temperature and top-k sampling
def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
            logits = logits[:, -1, :]
            if top_k is not None:
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1]
                logits = torch.where(
                    logits < min_val,
                    torch.tensor(float("-inf")).to(logits.device),
                    logits
                )
            if temperature > 0.0:
                logits = logits / temperature
                probabilities = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probabilities, num_samples=1)
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            if idx_next == eos_id:
                break
            idx = torch.cat((idx, idx_next), dim=1)
    return idx

torch.manual_seed(123)
token_ids = generate(
    model=model, 
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=15,
    context_size=GPT_CONFIG_124M['context_length'],
    temperature=1.4,
    top_k=25
)
print(token_ids_to_text(token_ids, tokenizer))

# Save the model
torch.save(model.state_dict(), "gpt_model.pth")

# Load the model
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(torch.load("gpt_model.pth", map_location=device))
model.eval()

# Save the model and optimizer state
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
}, "gpt_model_optimizer.pth")

# Load the model and optimizer state
checkpoint = torch.load("gpt_model_optimizer.pth", map_location=device)
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-1)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
model.train()
