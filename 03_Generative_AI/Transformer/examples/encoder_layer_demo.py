import torch

from src.models.encoder_layer import EncoderLayer


def main():

    encoder = EncoderLayer(
        d_model=512,
        num_heads=8,
        d_ff=2048,
    )

    x = torch.randn(
        2,
        5,
        512,
    )

    output = encoder(x)

    print("Input :", x.shape)
    print("Output:", output.shape)


if __name__ == "__main__":
    main()