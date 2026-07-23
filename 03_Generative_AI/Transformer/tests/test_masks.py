import torch

from src.utils.mask import (
    create_padding_mask,
    create_look_ahead_mask,
)


def test_padding_mask():

    sequence = torch.tensor([
        [5, 8, 2, 0, 0]
    ])

    mask = create_padding_mask(sequence)

    assert mask.shape == (
        1,
        1,
        1,
        5,
    )


def test_look_ahead_mask():

    mask = create_look_ahead_mask(5)

    assert mask.shape == (
        5,
        5,
    )