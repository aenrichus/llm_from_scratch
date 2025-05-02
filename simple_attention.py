import torch
inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your
     [0.55, 0.87, 0.66], # journey
     [0.57, 0.85, 0.64], # starts
     [0.22, 0.58, 0.33], # with
     [0.77, 0.25, 0.10], # one
     [0.05, 0.80, 0.55]] # step
)

query = inputs[1]
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

