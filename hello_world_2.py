import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# Hello, World! with device info
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Hello, World! Running on device: {device}")

# Simple tensor operation
a = torch.randn(3, 3, device=device)
b = torch.randn(3, 3, device=device)
start = time.time()
c = torch.mm(a, b)
end = time.time()
print("Random matrix multiplication result:")
print(c)
print(f"Computation took {(end - start)*1000:.2f} ms")

# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(9, 3)
        self.fc2 = nn.Linear(3, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

model = SimpleNet().to(device)
print("Model architecture:")
print(model)

# Forward pass with random input
input_tensor = torch.randn(1, 9, device=device)
output = model(input_tensor)
print("Model output for random input:")
print(output.item())