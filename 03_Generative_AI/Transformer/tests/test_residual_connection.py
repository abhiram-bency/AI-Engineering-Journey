import torch

from src.layers.residual_connection import ResidualConnection


def test_residual_connection_output_shape():

    batch_size = 2
    sequence_length = 5
    d_model = 512

    residual = ResidualConnection(d_model)

    x = torch.randn(
        batch_size,
        sequence_length,
        d_model,
    )

    sublayer_output = torch.randn(
        batch_size,
        sequence_length,
        d_model,
    )

    output = residual(
        x,
        sublayer_output,
    )

    assert output.shape == (
        batch_size,
        sequence_length,
        d_model,
    )