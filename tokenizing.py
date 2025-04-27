# Reading in a text file

with open('decay.txt', 'r', encoding='utf-8') as file:
    raw_text = file.read()

print('Total number of characters:', len(raw_text))
print(raw_text[:99])

# Simple tokenizer v1

import re

# 1) Preprocess into tokens
raw_text = open('decay.txt', 'r', encoding='utf-8').read()
pieces = re.split(r'([,.?_!"()\']|--|\s)', raw_text)
tokens = [t.strip() for t in pieces if t.strip()]

# 2) Build vocab from tokens
vocab = { token: idx 
          for idx, token in enumerate(sorted(set(tokens))) }

print("Vocab size    :", len(vocab))

# 3) Simple tokenizer class v1
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self, text):
        pieces = re.split(r'([,.?_!"()\']|--|\s)', text)
        tokens = [t.strip() for t in pieces if t.strip()]
        return [ self.str_to_int[t] for t in tokens ]

    def decode(self, ids):
        txt = " ".join(self.int_to_str[i] for i in ids)
        # glue punctuation back on
        return re.sub(r'\s+([,.?!"()\'])', r'\1', txt)

# Test
tokenizer = SimpleTokenizerV1(vocab)
ids = tokenizer.encode(raw_text)
print("First 20 tokens:", tokens[:20])
print("First 20 IDs   :", ids[:20])

# 4) Simple tokenizer class v2

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        preprocessed = [item if item in self.str_to_int
                        else "<|unk|>" for item in preprocessed]
        
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join(self.int_to_str[i] for i in ids)
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
    
# Test
tokenizer_v2 = SimpleTokenizerV2(vocab)
ids = tokenizer_v2.encode(raw_text)
print("First 20 tokens:", tokens[:20])
print("First 20 IDs   :", ids[:20])
