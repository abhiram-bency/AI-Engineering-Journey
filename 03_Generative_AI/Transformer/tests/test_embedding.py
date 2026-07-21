from src.embedding import TokenEmbedding
import torch

model = TokenEmbedding(
    vocab_size=1000,
    d_model=512,
)

tokens = torch.randint(
    0,
    1000,
    (2, 5),
)

output = model(tokens)

print(output.shape)