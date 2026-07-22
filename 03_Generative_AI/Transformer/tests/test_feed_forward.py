import torch

from src.layers.feed_forward import PositionwiseFeedForward


def test_feed_forward_output_shape():

    batch_size = 2
    sequence_length = 5
    d_model = 512
    d_ff = 2048

    ffn = PositionwiseFeedForward(
        d_model=d_model,
        d_ff=d_ff,
    )

    x = torch.randn(
        batch_size,
        sequence_length,
        d_model,
    )

    output = ffn(x)

    assert output.shape == (
        batch_size,
        sequence_length,
        d_model,
    )