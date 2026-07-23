import torch

from src.config import TransformerConfig
from src.models.decoder import Decoder


def test_decoder_output_shape():

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

    assert output.shape == (
        2,
        6,
        config.d_model,
    )