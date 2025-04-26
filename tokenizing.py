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

# 3) Simple tokenizer class
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self, text):
        pieces = re.split(r'([,.?_!"()\']|--|\s)', text)
        toks   = [t.strip() for t in pieces if t.strip()]
        return [ self.str_to_int[t] for t in toks ]

    def decode(self, ids):
        txt = " ".join(self.int_to_str[i] for i in ids)
        # glue punctuation back on
        return re.sub(r'\s+([,.?!"()\'])', r'\1', txt)

# 4) Test
tokenizer = SimpleTokenizerV1(vocab)
ids = tokenizer.encode(raw_text)
print("First 20 tokens:", tokens[:20])
print("First 20 IDs   :", ids[:20])
print("Vocab size    :", len(vocab))