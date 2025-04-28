import torch

input_ids = torch.tensor([2, 3, 6, 9])
vocabulary_size = 10
output_dim = 4

torch.manual_seed(0)
embedding_layer = torch.nn.Embedding(vocabulary_size, output_dim)
print(embedding_layer.weight)

print(embedding_layer(torch.tensor([2])))

print(embedding_layer(input_ids))
# The output of the embedding layer is a tensor of shape (4, 4) because we have 4 input IDs and the output dimension is 4.
