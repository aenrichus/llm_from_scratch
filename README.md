# Build an LLM from Scratch

Welcome to my personal learning project! 🎉

This repository documents my journey of **building a Large Language Model (LLM) from scratch** using PyTorch. The primary goal is educational — to deepen my understanding of how LLMs work under the hood, from tokenization to text generation.

## What's Inside

- 🧠 A custom GPT-style model built with PyTorch
- 🔤 Tokenization using `tiktoken` for GPT-2 compatibility
- 📈 Training and evaluation loops with loss tracking and token sampling
- 🔍 Support for top-k sampling and temperature scaling
- 🧪 Loss calculation and performance metrics
- 💾 Save/load functionality for models and optimizer states
- 🛠️ A simple text generation interface with context input
- 🧹 A fine-tuned version of the model for spam classification

## Why I'm Doing This

I've always been fascinated by the inner workings of LLMs. Rather than relying on high-level libraries, I wanted to **go deep into the nuts and bolts** of architecture, training, and inference — all from first principles.

## How to Run

1. Install dependencies:
   ```bash
   pip install torch tiktoken matplotlib
   ```

2. Prepare a training text file (`decay.txt`) in the project root.

3. Run the training script:
   ```bash
   python pretraining.py
   ```

## Notes

- This project is a **learning exercise**, not intended for production use.
- I’m actively iterating on this codebase as I explore new ideas and features. This includes experimenting with downstream tasks like spam classification.
- Feedback, suggestions, and collaboration are always welcome!

## License

This project is open source and available under the MIT License.
