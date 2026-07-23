import torch

from src.models.decoder_layer import DecoderLayer


def main():

    decoder = DecoderLayer(
        d_model=512,
        num_heads=8,
        d_ff=2048,
    )

    decoder_input = torch.randn(
        2,
        6,
        512,
    )

    encoder_output = torch.randn(
        2,
        8,
        512,
    )

    output = decoder(
        x=decoder_input,
        encoder_output=encoder_output,
    )

    print("Decoder Input Shape :", decoder_input.shape)
    print("Encoder Output Shape:", encoder_output.shape)
    print("Decoder Output Shape:", output.shape)


if __name__ == "__main__":
    main()