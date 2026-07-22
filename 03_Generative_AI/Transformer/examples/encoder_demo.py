import torch

from src.config import TransformerConfig
from src.models.encoder import Encoder


def main():

    config = TransformerConfig()

    encoder = Encoder(config)

    x = torch.randn(
        2,
        5,
        config.d_model,
    )

    output = encoder(x)

    print("Input :", x.shape)
    print("Output:", output.shape)
    print("Number of Encoder Layers:", len(encoder.layers))


if __name__ == "__main__":
    main()