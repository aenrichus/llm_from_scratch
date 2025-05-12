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
print(f"Example entry: \n {data[50]}")

