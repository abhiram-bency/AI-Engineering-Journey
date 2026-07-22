import torch

from src.config import TransformerConfig
from src.models.encoder import Encoder


def test_encoder_output_shape():

    config = TransformerConfig()

    encoder = Encoder(config)

    x = torch.randn(
        2,
        5,
        config.d_model,
    )

    output = encoder(x)

    assert output.shape == (
        2,
        5,
        config.d_model,
    )