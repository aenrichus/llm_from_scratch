import torch, tiktoken
from torch.utils.data import Dataset, DataLoader

# This code is a PyTorch dataset and dataloader for processing text data using the GPT-2 tokenizer.
# It tokenizes the text, creates input-target pairs, and prepares them for training a language model.
# The dataset is designed to handle large text files by splitting them into smaller chunks based on the specified max_length and stride.
# The dataloader can be customized with various parameters like batch size, shuffling, and number of workers for parallel data loading.

class GPTDatasetV1(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text, allowed_special="all")

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(text, batch_size = 4, max_length = 256, stride = 128, shuffle = True, drop_last = True, num_workers = 0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(text, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers)
    
    return dataloader

# Test the dataloader
with open('decay.txt', 'r', encoding='utf-8') as file:
    raw_text = file.read()

dataloader = create_dataloader_v1(raw_text, batch_size=4, max_length=4, stride=4, shuffle=False)
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)

second_batch = next(data_iter)
print(second_batch)

# Create token embedding layer
# The token embedding layer is used to convert token IDs into dense vectors of a specified output dimension.
# This is typically the first layer in a transformer model.
vocabulary_size = 50257  # GPT-2 vocabulary size
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocabulary_size, output_dim)

max_length = 4
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)

token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)

# Create position embedding layer
# The position embedding layer is used to add positional information to the token embeddings.
# This is important for transformer models as they do not have a built-in notion of sequence order.
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)

input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)

