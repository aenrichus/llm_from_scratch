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