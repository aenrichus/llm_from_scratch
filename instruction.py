# Instruction Tuning

# Download the dataset
import json
import os
import urllib.request

def download_and_load_file(file_path, url):
    """
    Download a file from a URL and save it to the specified file path.
    """
    if not os.path.exists(file_path):
        print(f"Downloading {url} to {file_path}...")
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode('utf-8')
        with open(file_path, 'w', encoding="utf-8") as file:
                file.write(text_data)
        print("Download complete.")
    else:
        print(f"{file_path} already exists. Skipping download.")
        with open(file_path, 'r', encoding="utf-8") as file:
            text_data = file.read()
    
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    return data

file_path = "instruction-data.json"
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"

data = download_and_load_file(file_path, url)
print(f"Loaded {len(data)} instruction-response pairs.")
print("Example entry: \n", data[50])
print("Another entry: \n", data[999])

# Implement Alpaca prompt formatting
def format_input(entry):
    instruction_text = (
          f"Below is an instruction that describes a task. "
          f"Write a response that appropriately completes the request.\n\n"
          f"### Instruction:\n{entry['instruction']}\n\n"
    )
    
    input_text = (f"### Input:\n{entry['input']}\n\n" if entry['input'] else "")

    return instruction_text + input_text

model_input = format_input(data[50])
desired_response = f"### Response:\n{data[50]['output']}\n\n"
print(model_input + desired_response)

model_input = format_input(data[999])
desired_response = f"### Response:\n{data[999]['output']}\n\n"
print(model_input + desired_response)

# Partition the dataset for training, validation, and testing
train_portion = int(len(data) * 0.85)
test_portion = int(len(data) * 0.10)
val_portion = len(data) - train_portion - test_portion

train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]
val_data = data[train_portion + test_portion:]

print(f"Training data size: {len(train_data)}")
print(f"Validation data size: {len(val_data)}")
print(f"Testing data size: {len(test_data)}")

# Implement instruction dataset class
import torch
from torch.utils.data import Dataset

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in data:
            instuction_plus_input = format_input(entry)
            response_text = f"### Response:\n{entry['output']}\n\n"
            full_text = instuction_plus_input + response_text
            self.encoded_texts.append(tokenizer.encode(full_text))

    def __getitem__(self, idx):
        return self.encoded_texts[idx]
    
    def __len__(self):
        return len(self.data)
    
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))

# Create custom collate function
def custom_collate_v1(batch, pad_token_id=50256, device="cpu"):
    batch_max_len = max(len(item)+1 for item in batch)
    inputs_list = []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = (new_item + [pad_token_id] * (batch_max_len - len(new_item)))
        inputs = torch.tensor(padded[:-1])
        inputs_list.append(inputs)
    
    inputs_tensor = torch.stack(inputs_list).to(device)
    return inputs_tensor

inputs_1 = [0, 1, 2, 3, 4]
inputs_2 = [5, 6]
inputs_3 = [7, 8, 9]

batch = [inputs_1, inputs_2, inputs_3]
print(custom_collate_v1(batch))

def custom_collate_v2(batch, pad_token_id=50256, device="cpu"):
    batch_max_len = max(len(item)+1 for item in batch)
    inputs_list, targets_list = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id] * (batch_max_len - len(new_item))

        padded = (new_item + [pad_token_id] * (batch_max_len - len(new_item)))
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])
        inputs_list.append(inputs)
        targets_list.append(targets)
    
    inputs_tensor = torch.stack(inputs_list).to(device)
    targets_tensor = torch.stack(targets_list).to(device)

    return inputs_tensor, targets_tensor

inputs, targets = custom_collate_v2(batch)
print(inputs)
print(targets)

def custom_collate_fn(batch, pad_token_id=50256, ignore_index=-100, allowed_max_len=None, device="cpu"):
    """
    Custom collate function for DataLoader.
    """
    batch_max_len = max(len(item)+1 for item in batch)
    inputs_list, targets_list = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = (new_item + [pad_token_id] * (batch_max_len - len(new_item)))
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index
        
        if allowed_max_len is not None:
            inputs = inputs[:allowed_max_len]
            targets = targets[:allowed_max_len]

        inputs_list.append(inputs)
        targets_list.append(targets)
    
    inputs_tensor = torch.stack(inputs_list).to(device)
    targets_tensor = torch.stack(targets_list).to(device)

    return inputs_tensor, targets_tensor

inputs, targets = custom_collate_fn(batch)
print(inputs)
print(targets)

logits_1 = torch.tensor(
    [[-1.0, 1.0],
     [-0.5, 1.5]]
)
targets_1 = torch.tensor([0, 1])
loss_1 = torch.nn.functional.cross_entropy(logits_1, targets_1)
print(loss_1)

logits_2 = torch.tensor(
    [[-1.0, 1.0],
     [-0.5, 1.5],
     [-0.5, 1.5]]
)
targets_2 = torch.tensor([0, 1, 1])
loss_2 = torch.nn.functional.cross_entropy(logits_2, targets_2)
print(loss_2)

targets_3 = torch.tensor([0, 1, -100])
loss_3 = torch.nn.functional.cross_entropy(logits_2, targets_3)
print(loss_3)
print("loss_1 == loss_3: ", loss_1 == loss_3)

device = torch.device("mps" if torch.mps.is_available() else "cpu")
print("Device: ", device)

from functools import partial

custom_collate_fn = partial(
    custom_collate_fn,
    allowed_max_len=1024,
    device=device
)

# Initialize data loaders
from torch.utils.data import DataLoader

num_workers = 0
batch_size = 8

torch.manual_seed(123)

train_dataset = InstructionDataset(train_data, tokenizer)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=custom_collate_fn,
    num_workers=num_workers,
    drop_last=True
)
val_dataset = InstructionDataset(val_data, tokenizer)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=custom_collate_fn,
    num_workers=num_workers,
    drop_last=False
)
test_dataset = InstructionDataset(test_data, tokenizer)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=custom_collate_fn,
    num_workers=num_workers,
    drop_last=False
)

print("Train loader:")
for inputs, targets in train_dataloader:
    print(inputs.shape, targets.shape)

# Load pretrained model
from gpt_download import download_and_load_gpt2
from gpt_model import GPTModel
from load_pretrained import load_weights_into_gpt

BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True
}
model_configs = {
    "gpt2-small (124M)": {
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12
    },
    "gpt2-medium (355M)": {
        "emb_dim": 1024,
        "n_heads": 16,
        "n_layers": 24
    },
    "gpt2-large (774M)": {
        "emb_dim": 1280,
        "n_heads": 20,
        "n_layers": 36
    },
    "gpt2-xl (1558M)": {
        "emb_dim": 1600,
        "n_heads": 25,
        "n_layers": 48
    }
}
CHOOSE_MODEL = "gpt2-medium (355M)"
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

settings, params = download_and_load_gpt2(
    model_size=model_size,
    models_dir="gpt2"
)

model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()

torch.manual_seed(123)
input_text = format_input(val_data[0])
print(input_text)

from pretraining import generate, text_to_token_ids, token_ids_to_text

token_ids = generate(
    model=model,
    idx=text_to_token_ids(input_text, tokenizer),
    max_new_tokens=35,
    context_size=BASE_CONFIG["context_length"],
    eos_id=50256,
)
generated_text = token_ids_to_text(token_ids, tokenizer)

response_text = generated_text[len(input_text):].strip()
print(response_text)

# Finetune on instruction data
from pretraining import calc_loss_loader, train_model_simple

model.to(device)
torch.manual_seed(123)

with torch.no_grad():
    train_loss = calc_loss_loader(train_dataloader, model, device, num_batches=5)
    val_loss = calc_loss_loader(val_dataloader, model, device, num_batches=5)

print(f"Train loss: {train_loss:.4f}")
print(f"Validation loss: {val_loss:.4f}")

import time

start_time = time.time()
torch.manual_seed(123)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)
num_epochs = 5

train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_dataloader, val_dataloader, optimizer, device, num_epochs=num_epochs, 
    eval_freq=5, eval_iter=5, start_context=format_input(val_data[0]), tokenizer=tokenizer
)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Execution time: {execution_time_minutes:.2f} minutes")

from pretraining import plot_losses
epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

# Extract and save responses
torch.manual_seed(123)

for entry in test_data[:3]:
    input_text = format_input(entry)
    print(input_text)
    
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256,
    )
    
    generated_text = token_ids_to_text(token_ids, tokenizer)
    response_text = generated_text[len(input_text):].replace("### Response:", "").strip()
    print(f"\nCorrect response:\n{entry['output']}")
    print(f"Generated response:\n{response_text.strip()}")
    print("-" * 50)

# Generate test set responses
from tqdm import tqdm

for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
    input_text = format_input(entry)
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256
    )
    
    generated_text = token_ids_to_text(token_ids, tokenizer)
    # response_text = generated_text[len(input_text):].replace("### Response:", "").strip()
    # Try to find where the response actually starts
    split_marker = "### Response:"
    if split_marker in generated_text:
        response_text = generated_text.split(split_marker, 1)[-1].strip()
    else:
        response_text = generated_text.strip()

    test_data[i]["generated_response"] = response_text

with open("instruction-data-with-response.json", "w") as file:
    json.dump(test_data, file, indent=4)

print(test_data[0])

import re

file_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL)}_sft.pth"
torch.save(model.state_dict(), file_name)
print(f"Model saved to {file_name}")

# Ensure ollama is running
import psutil

def check_if_running(process_name):
    running = False
    for proc in psutil.process_iter(["name"]):
        if process_name in proc.info["name"]:
            running = True
            break
    return running

ollama_running = check_if_running("ollama")

if not ollama_running:
    raise RuntimeError("Ollama not running. Launch ollama before proceeding.")
print("Ollama running: ", check_if_running("ollama"))

# Query local ollama model
import urllib.request

def query_model(
        prompt,
        model="llama3",
        url="http://localhost:11434/api/chat"
):
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "options": {
            "seed": 123,
            "temperature": 0,
            "num_ctx": 2048
        }
    }

    payload = json.dumps(data).encode("utf-8")
    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("Content-Type", "application/json")

    response_data = ""
    with urllib.request.urlopen(request) as response:
        while True:
            line = response.readline().decode("utf-8")
            if not line:
                break
            response_json = json.loads(line)
            response_data += response_json["message"]["content"]
    
    return response_data

model = "llama3"
result = query_model("What do Llamas eat?", model)
print(result)

# Test the model with a few examples
for entry in test_data[:3]:
    prompt = (
        f"Given the WolfGPT model input `{format_input(entry)}` "
        f"and the correct output `{entry['output']}`, "
        f"score the WolfGPT model response `{entry['generated_response']}`"
        f" on a scale from 0 to 100, where 100 is the best score. "
        f"Score only WolfGPT model response, not your own response."
    )
    print("\nDataset response:")
    print(">>", entry["output"])
    print("\nModel response:")
    print(">>", entry["generated_response"])
    print("\nScore:")
    print(">>", query_model(prompt))
    print("\n", "-" * 50)

# Evaluate instruction finetuning model
def generate_model_scores(json_data, json_key, model="llama3"):
    scores = []
    for entry in tqdm(json_data, desc="Generating scores"):
        prompt = (
            f"Given the WolfGPT model input `{format_input(entry)}` "
            f"and the correct output `{entry['output']}`, "
            f"score the WolfGPT model response `{entry[json_key]}` "
            f"on a scale from 0 to 100, where 100 is the best score. "
            f"Score only WolfGPT model response, not your own response. "
            f"Respond with the integer number only."
        )
        score = query_model(prompt, model)
        try:
            scores.append(int(score))
        except ValueError:
            print(f"Error converting score: {score}")
            continue

    return scores

scores = generate_model_scores(test_data, "generated_response")
print(f"Number of scores: {len(scores)} of {len(test_data)}")
print(f"Average score: {sum(scores) / len(scores):.2f}\n")
