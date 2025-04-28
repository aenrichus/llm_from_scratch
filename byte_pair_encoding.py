from importlib.metadata import version
import tiktoken
print(f"tiktoken version: {version('tiktoken')}")

tokenizer = tiktoken.get_encoding("gpt2")

with open('decay.txt', 'r', encoding='utf-8') as file:
    raw_text = file.read()

integers = tokenizer.encode(raw_text, allowed_special="all")
print("First 20 tokens:", integers[:20])
print(len(integers))

strings = tokenizer.decode(integers)
print("First 20 tokens:", strings[:200])
print(len(strings))



