import torch

from src.layers.layer_normalization import LayerNormalization


def test_layer_normalization_output_shape():

    batch_size = 2
    sequence_length = 5
    d_model = 512

    layer_norm = LayerNormalization(
        d_model=d_model,
    )

    x = torch.randn(
        batch_size,
        sequence_length,
        d_model,
    )

    output = layer_norm(x)

    assert output.shape == (
        batch_size,
        sequence_length,
        d_model,
    )