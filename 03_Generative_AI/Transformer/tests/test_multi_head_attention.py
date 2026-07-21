import torch

from src.attention.multi_head_attention import MultiHeadAttention


def test_multi_head_attention():

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

    output, weights = attention(
        query=x,
        key=x,
        value=x,
    )

    print("Input Shape      :", x.shape)
    print("Output Shape     :", output.shape)
    print("Attention Shape  :", weights.shape)


if __name__ == "__main__":
    test_multi_head_attention()