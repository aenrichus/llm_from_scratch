import urllib.request
import zipfile
import os
from pathlib import Path

url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "sms_spam_collection.zip"
extract_path = "sms_spam_collection"
data_file_path = Path(extract_path) / "SMSSpamCollection.tsv"

def download_and_extract_data(url, zip_path, extract_path, data_file_path):
    if data_file_path.exists():
        print(f"Data file already exists at {data_file_path}.")
        return
    
    with urllib.request.urlopen(url) as response:
        with open(zip_path, 'wb') as out_file:
            out_file.write(response.read())

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    
    original_file_path = Path(extract_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"Data file extracted to {data_file_path}.")

download_and_extract_data(url, zip_path, extract_path, data_file_path)

import pandas as pd
df = pd.read_csv(data_file_path, sep='\t', header=None, names=['label', 'message'])
print(df)
print(df["label"].value_counts())

# Create a balanced dataset
def create_balanced_dataset(df):
    num_spam = df[df['label'] == 'spam'].shape[0]
    ham_subset = df[df['label'] == 'ham'].sample(num_spam, random_state=123)
    balanced_df = pd.concat([df[df['label'] == 'spam'], ham_subset])
    return balanced_df

balanced_df = create_balanced_dataset(df)
print(balanced_df["label"].value_counts())

balanced_df["label"] = balanced_df["label"].map({"ham": 0, "spam": 1})

# Split the dataset into training, validation, and test sets
def random_split(df, train_size, val_size):
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    train_end = int(train_size * int(len(df)))
    val_end = train_end + int(val_size * int(len(df)))

    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]

    return train_df, val_df, test_df

train_df, val_df, test_df = random_split(balanced_df, 0.7, 0.1)

# Save the datasets to CSV files
train_df.to_csv("train.csv", index=None)
val_df.to_csv("val.csv", index=None)
test_df.to_csv("test.csv", index=None)

# Create data loaders
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))

# Set up pytorch dataset class
import torch
from torch.utils.data import Dataset

class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)
        self.encoded_texts = [tokenizer.encode(text) for text in self.data['message']]
        
        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            self.encoded_texts = [encoded_text[:self.max_length] for encoded_text in self.encoded_texts]

        self.encoded_texts = [encoded_text + [pad_token_id] * (self.max_length - len(encoded_text)) for encoded_text in self.encoded_texts]

    def __getitem__(self, idx):
        encoded = self.encoded_texts[idx]
        label = self.data.iloc[idx]['label']
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long)
    
    def __len__(self):
        return len(self.data)
    
    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length
    
train_dataset = SpamDataset(
    csv_file="train.csv",
    max_length=None,
    tokenizer=tokenizer
)

print(train_dataset.max_length)

val_dataset = SpamDataset(
    csv_file="val.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)
test_dataset = SpamDataset(
    csv_file="test.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)

# Create pytorch dataloaders
from torch.utils.data import DataLoader

num_workers = 0 # Use 0 for higher compatibility, but use up to 8 for for faster on M1 Max MBP w/ 64GB RAM
batch_size = 8

## Note: This runs the read multiple times when you add workers, and makes it so you cannot use 0 workers
## Fix: Wrap the read into the if __name__ == '__main__' block
# if __name__ == '__main__': # Required for multiprocessing on MacOS
torch.manual_seed(123)

train_loader = DataLoader(
    dataset = train_dataset,
    batch_size = batch_size,
    shuffle = True,
    num_workers = num_workers,
    # persistent_workers=True, # Keep workers alive for multiple epochs
    # prefetch_factor=None, # Number of batches to prefetch
    drop_last = True
)
val_loader = DataLoader(
    dataset = val_dataset,
    batch_size = batch_size,
    shuffle = False,
    num_workers = num_workers,
    # persistent_workers=True, # Keep workers alive for multiple epochs
    # prefetch_factor=None, # Number of batches to prefetch
    drop_last = False
)
test_loader = DataLoader(
    dataset = test_dataset,
    batch_size = batch_size,
    shuffle = False,
    num_workers = num_workers,
    # persistent_workers=True, # Keep workers alive for multiple epochs
    # prefetch_factor=None, # Number of batches to prefetch
    drop_last = False
)

for input_batch, target_batch in train_loader:
    pass
print(input_batch.shape)
print(target_batch.shape)

# Check the number of batches in each dataloader
print(len(train_loader))
print(len(val_loader))
print(len(test_loader))

# Initialize the pretrained model
CHOOSE_MODEL = "gpt2-small (124M)"
INPUT_PROMPT = "This is a spam message"
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
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

# Load the pretrained model
from gpt_download import download_and_load_gpt2
from gpt_model import GPTModel
from load_pretrained import load_weights_into_gpt

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()

# Test the model with a simple text generation
from gpt_model import generate_text_simple
from pretraining import text_to_token_ids, token_ids_to_text

text_1 = "Every effort moves you"
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(text_1, tokenizer),
    max_new_tokens=15,
    context_size=BASE_CONFIG["context_length"]
)
print(token_ids_to_text(token_ids, tokenizer))

# Test the model at spam classification
text_2 = ("Is the following text 'spam'? Answer with 'yes' or 'no':"
          " 'You are a winner and you have been specially"
          " selected to receive $1000 cash or a $2000 award.'")
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(text_2, tokenizer),
    max_new_tokens=23,
    context_size=BASE_CONFIG["context_length"]
)
print(token_ids_to_text(token_ids, tokenizer))

# Look at the model architecture
# print(model)

# Add a classification head to the model
import torch.nn as nn
torch.manual_seed(123)
num_classes = 2
model.out_head = nn.Linear(
    in_features=BASE_CONFIG["emb_dim"],
    out_features=num_classes
)

# Make the relevant parameters trainable
for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True
for param in model.final_norm.parameters():
    param.requires_grad = True

inputs = tokenizer.encode("Do you have time")
inputs = torch.tensor(inputs).unsqueeze(0)
print(inputs)
print(inputs.shape)

with torch.no_grad():
    outputs = model(inputs)
print(outputs)
print(outputs.shape)

print(outputs[:, -1, :])

# Calculate classification loss and accuracy
probas = torch.softmax(outputs[:, -1, :], dim=-1)
label = torch.argmax(probas)
print(label.item())

# Softmax not needed for argmax
logits = outputs[:, -1, :]
label = torch.argmax(logits)
print(label.item())

## Calculate accuracy
def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            with torch.no_grad():
                outputs = model(input_batch)
            
            logits = outputs[:, -1, :]
            predictions = torch.argmax(logits, dim=-1)

            num_examples += predictions.shape[0]
            correct_predictions += (predictions == target_batch).sum().item()
        else:
            break
    
    return correct_predictions / num_examples

device = torch.device("mps" if torch.mps.is_available() else "cpu")
model.to(device)

torch.manual_seed(123)
train_accuracy = calc_accuracy_loader(
    data_loader=train_loader,
    model=model,
    device=device,
    num_batches=10
)
val_accuracy = calc_accuracy_loader(
    data_loader=val_loader,
    model=model,
    device=device,
    num_batches=10
)
test_accuracy = calc_accuracy_loader(
    data_loader=test_loader,
    model=model,
    device=device,
    num_batches=10
)
print(f"Train accuracy: {train_accuracy*100:.2f}")
print(f"Validation accuracy: {val_accuracy*100:.2f}")
print(f"Test accuracy: {test_accuracy*100:.2f}")

## Calculate batch loss
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    outputs = model(input_batch)
    logits = outputs[:, -1, :]
    loss = nn.functional.cross_entropy(logits, target_batch)
    
    return loss

## Calculate classification loss
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
    
    return total_loss / num_batches

with torch.no_grad():
    train_loss = calc_loss_loader(
        data_loader=train_loader,
        model=model,
        device=device,
        num_batches=5
    )
    val_loss = calc_loss_loader(
        data_loader=val_loader,
        model=model,
        device=device,
        num_batches=5
    )
    test_loss = calc_loss_loader(
        data_loader=test_loader,
        model=model,
        device=device,
        num_batches=5
    )
print(f"Train loss: {train_loss:.4f}")
print(f"Validation loss: {val_loss:.4f}")
print(f"Test loss: {test_loss:.4f}")

# Finetune the model for spam classification
def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            examples_seen += input_batch.shape[0]
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Epoch {epoch+1}/{num_epochs}, Step {global_step:06d}, "
                      f"Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}, ")
                
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)

        print(f"Train Accuracy: {train_accuracy*100:.2f}%, ", end="")
        print(f"Validation Accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)
    
    return train_losses, val_losses, train_accs, val_accs, examples_seen

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

import time

start_time = time.time()
torch.manual_seed(123)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-5,
    weight_decay=0.01
)
num_epochs = 5

train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=50, eval_iter=5
)

end_time = time.time()
execution_time_min = (end_time - start_time) / 60
print(f"Execution time: {execution_time_min:.2f} minutes")

# Plot the classification loss
import matplotlib.pyplot as plt

def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    ax2 = ax1.twiny()
    ax2.plot(examples_seen, train_values, alpha=0)
    ax2.set_xlabel("Examples seen")

    fig.tight_layout()
    plt.savefig(f"spam_classification_{label}.pdf")
    plt.show()

## Plot loss
epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
examples_tensor = torch.linspace(0, examples_seen, len(train_losses))
plot_values(epochs_tensor, examples_tensor, train_losses, val_losses)

## Plot accuracy
epochs_tensor = torch.linspace(0, num_epochs, len(train_accs))
examples_tensor = torch.linspace(0, examples_seen, len(train_accs))
plot_values(epochs_tensor, examples_tensor, train_accs, val_accs, label="accuracy")

train_accuracy = calc_accuracy_loader(train_loader, model, device)
val_accuracy = calc_accuracy_loader(val_loader, model, device)
test_accuracy = calc_accuracy_loader(test_loader, model, device)
print(f"Final Train accuracy: {train_accuracy*100:.2f}")
print(f"Final Validation accuracy: {val_accuracy*100:.2f}")
print(f"Final Test accuracy: {test_accuracy*100:.2f}")

# Use the model to classify new messages
def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    model.eval()

    encoded_text = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[1]

    encoded_text += [pad_token_id] * (max_length - len(encoded_text))

    input_tensor = torch.tensor(encoded_text, device=device).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)[::, -1, :]
    
    predicted_label = torch.argmax(logits, dim=-1).item()
    
    return "spam" if predicted_label == 1 else "ham"

text_1 = "You are a winner you have been specially selected to receive $1000 cash or a $2000 award."
print(classify_review(text_1, model, tokenizer, device, max_length=train_dataset.max_length))
text_2 = "Hey, just wanted to check if we're still on for dinner tonight? Let me know!"
print(classify_review(text_2, model, tokenizer, device, max_length=train_dataset.max_length))

# Save the model
torch.save(model.state_dict(), "spam_classifier.pth")

# Load the model
model_state_dict = torch.load("spam_classifier.pth", map_location=device)
model.load_state_dict(model_state_dict)

print("Model loaded successfully.")