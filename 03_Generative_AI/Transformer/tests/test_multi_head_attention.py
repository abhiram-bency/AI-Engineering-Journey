import torch

from src.attention.multi_head_attention import MultiHeadAttention


def test_split_heads():

    batch_size = 2
    sequence_length = 5
    d_model = 512
    num_heads = 8

    attention = MultiHeadAttention(
        d_model=d_model,
        num_heads=num_heads,
    )

    x = torch.randn(
        batch_size,
        sequence_length,
        d_model,
    )

    output = attention._split_heads(x)

    print(output.shape)


if __name__ == "__main__":
    test_split_heads()