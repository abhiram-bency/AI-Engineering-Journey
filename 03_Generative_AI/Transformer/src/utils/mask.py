"""
=========================================================
Attention Mask Utilities
=========================================================

Provides helper functions for creating
padding masks and causal masks used by
the Transformer architecture.
"""

import torch
from torch import Tensor


def create_padding_mask(
    sequence: Tensor,
    pad_token: int = 0,
) -> Tensor:
    """
    Creates a padding mask.

    Shape:
        (batch, 1, 1, sequence_length)
    """

    return (sequence != pad_token).unsqueeze(1).unsqueeze(2)


def create_look_ahead_mask(
    size: int,
) -> Tensor:
    """
    Creates a causal mask.

    Shape:
        (1, size, size)
    """

    mask = torch.triu(
        torch.ones(size, size),
        diagonal=1,
    )

    return mask == 0