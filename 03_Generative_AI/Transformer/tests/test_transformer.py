import torch

from src.config import TransformerConfig
from src.models.transformer import Transformer


def test_transformer_output_shape():

    config = TransformerConfig()

    transformer = Transformer(config)

    source = torch.randint(
        0,
        config.vocab_size,
        (2, 8),
    )

    target = torch.randint(
        0,
        config.vocab_size,
        (2, 6),
    )

    output = transformer(
        source,
        target,
    )

    assert output.shape == (
        2,
        6,
        config.vocab_size,
    )