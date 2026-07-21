"""
=========================================================
Multi-Head Attention
=========================================================

Project:
    AI Engineering Journey

Module:
    Transformer from Scratch using PyTorch

Description:
    Implements Multi-Head Attention as introduced in
    "Attention Is All You Need".

Reference:
    Vaswani et al., 2017
"""

import torch.nn as nn
from torch import Tensor

from .scaled_dot_product import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module.

    Args:
        d_model:
            Embedding dimension.

        num_heads:
            Number of attention heads.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
    ) -> None:
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads."
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dimension = d_model // num_heads

        self.query_projection = nn.Linear(
            d_model,
            d_model,
        )

        self.key_projection = nn.Linear(
            d_model,
            d_model,
        )

        self.value_projection = nn.Linear(
            d_model,
            d_model,
        )

        self.output_projection = nn.Linear(
            d_model,
            d_model,
        )

        self.attention = ScaledDotProductAttention()