import torch

from src.models.decoder_layer import DecoderLayer


def test_decoder_layer_output_shape():

    batch_size = 2
    target_sequence_length = 6
    source_sequence_length = 8

    d_model = 512
    num_heads = 8
    d_ff = 2048

    decoder = DecoderLayer(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
    )

    decoder_input = torch.randn(
        batch_size,
        target_sequence_length,
        d_model,
    )

    encoder_output = torch.randn(
        batch_size,
        source_sequence_length,
        d_model,
    )

    output = decoder(
        x=decoder_input,
        encoder_output=encoder_output,
    )

    assert output.shape == (
        batch_size,
        target_sequence_length,
        d_model,
    )