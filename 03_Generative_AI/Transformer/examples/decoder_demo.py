import torch

from src.config import TransformerConfig
from src.models.decoder import Decoder


def main():

    config = TransformerConfig()

    decoder = Decoder(config)

    decoder_input = torch.randn(
        2,
        6,
        config.d_model,
    )

    encoder_output = torch.randn(
        2,
        8,
        config.d_model,
    )

    output = decoder(
        x=decoder_input,
        encoder_output=encoder_output,
    )

    print("Decoder Input :", decoder_input.shape)
    print("Encoder Output:", encoder_output.shape)
    print("Decoder Output:", output.shape)
    print("Number of Decoder Layers:", len(decoder.layers))


if __name__ == "__main__":
    main()