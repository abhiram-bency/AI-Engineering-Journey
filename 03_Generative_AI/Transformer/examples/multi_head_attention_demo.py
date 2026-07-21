import torch

from src.attention.multi_head_attention import MultiHeadAttention


def main():

    attention = MultiHeadAttention(
        d_model=512,
        num_heads=8,
    )

    x = torch.randn(2, 5, 512)

    output, weights = attention(
        query=x,
        key=x,
        value=x,
    )

    print("Input:", x.shape)
    print("Output:", output.shape)
    print("Weights:", weights.shape)


if __name__ == "__main__":
    main()