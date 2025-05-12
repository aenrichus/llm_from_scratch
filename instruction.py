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

