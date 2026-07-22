import torch

from src.attention.multi_head_attention import MultiHeadAttention


def test_multi_head_attention_output_shape():

    batch_size = 2
    sequence_length = 5
    d_model = 512
    num_heads = 8

    attention = MultiHeadAttention(
        d_model=d_model,
        num_heads=num_heads,
    )

    x = torch.randn(
        batch_size,
        sequence_length,
        d_model,
    )

    output, weights = attention(
        query=x,
        key=x,
        value=x,
    )

    assert output.shape == (
        batch_size,
        sequence_length,
        d_model,
    )

    assert weights.shape == (
        batch_size,
        num_heads,
        sequence_length,
        sequence_length,
    )


import pytest

def test_invalid_head_configuration():

    with pytest.raises(ValueError):

        MultiHeadAttention(
            d_model=510,
            num_heads=8,
        )

