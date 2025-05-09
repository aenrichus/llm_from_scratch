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

num_workers = 1 # Use 0 for higher compatibility, but use up to 8 for for faster on M1 Max MBP w/ 64GB RAM
batch_size = 8

## Note: This runs the read multiple times when you add workers, and makes it so you cannot use 0 workers
## Fix: Wrap the read into the if __name__ == '__main__' block
if __name__ == '__main__': # Required for multiprocessing on MacOS
    torch.manual_seed(123)

    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = num_workers,
        persistent_workers=True, # Keep workers alive for multiple epochs
        prefetch_factor=None, # Number of batches to prefetch
        drop_last = True
    )
    val_loader = DataLoader(
        dataset = val_dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = num_workers,
        persistent_workers=True, # Keep workers alive for multiple epochs
        prefetch_factor=None, # Number of batches to prefetch
        drop_last = False
    )
    test_loader = DataLoader(
        dataset = test_dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = num_workers,
        persistent_workers=True, # Keep workers alive for multiple epochs
        prefetch_factor=None, # Number of batches to prefetch
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



