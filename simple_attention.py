import torch
inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your
     [0.55, 0.87, 0.66], # journey
     [0.57, 0.85, 0.64], # starts
     [0.22, 0.58, 0.33], # with
     [0.77, 0.25, 0.10], # one
     [0.05, 0.80, 0.55]] # step
)

# Calculate the attention scores (a, dot product)
query = inputs[1] # (x)
attn_scores_2 = torch.empty(inputs.shape[0])

for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)
print(attn_scores_2)

# Normalize the attention scores (simple)
attn_weights_2_tmp = attn_scores_2 / torch.sum(attn_scores_2)
print(attn_weights_2_tmp)
print(torch.sum(attn_weights_2_tmp))

# Normalize the attention scores (softmax)
def softmax_naive(x):
    return torch.exp(x) / torch.sum(torch.exp(x), dim=0)

atten_weights_2_naive = softmax_naive(attn_scores_2)
print(atten_weights_2_naive)
print(torch.sum(atten_weights_2_naive))

# Normalize with torch softmax
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print(attn_weights_2)
print(torch.sum(attn_weights_2))

# Calculate the context vector (z)
context_vector_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vector_2 += attn_weights_2[i] * x_i
print(context_vector_2)

# Compute attention scores for all input tokens
attn_scores = torch.empty(6, 6)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i][j] = torch.dot(x_i, x_j)
print(attn_scores)

# Computer attention scores with matrix multiplication
attn_scores = torch.matmul(inputs, inputs.T)
print(attn_scores)

# Normalize the attention scores
attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)

row_2_sum = torch.sum(attn_weights[1])
print(row_2_sum)
print(torch.sum(attn_weights, dim=-1))

context_vectors = torch.matmul(attn_weights, inputs)
print(context_vectors)
print(context_vector_2)