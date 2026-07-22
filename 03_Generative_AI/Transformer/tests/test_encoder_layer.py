import torch

from src.models.encoder_layer import EncoderLayer


def test_encoder_layer_output_shape():

    batch_size = 2
    sequence_length = 5

    d_model = 512
    num_heads = 8
    d_ff = 2048

    encoder = EncoderLayer(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
    )

    x = torch.randn(
        batch_size,
        sequence_length,
        d_model,
    )

    output = encoder(x)

    assert output.shape == (
        batch_size,
        sequence_length,
        d_model,
    )