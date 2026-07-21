import torch

from src.attention.scaled_dot_product import (
    ScaledDotProductAttention,
)

batch_size = 2
sequence_length = 5
d_model = 64

query = torch.randn(
    batch_size,
    sequence_length,
    d_model,
)

key = torch.randn(
    batch_size,
    sequence_length,
    d_model,
)

value = torch.randn(
    batch_size,
    sequence_length,
    d_model,
)

attention = ScaledDotProductAttention()

output, weights = attention(
    query,
    key,
    value,
)

print(output.shape)
print(weights.shape)