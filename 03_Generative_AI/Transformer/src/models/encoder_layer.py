"""
=========================================================
Transformer Encoder Layer
=========================================================

Project:
    AI Engineering Journey

Module:
    Transformer from Scratch using PyTorch

Description:
    Implements a single Transformer Encoder Layer as
    described in "Attention Is All You Need".

Reference:
    Vaswani et al., 2017
"""

import torch.nn as nn
from torch import Tensor

from src.attention.multi_head_attention import MultiHeadAttention
from src.layers.feed_forward import PositionwiseFeedForward
from src.layers.residual_connection import ResidualConnection


class EncoderLayer(nn.Module):
    """
    Single Transformer Encoder Layer.

    Consists of:

    1. Multi-Head Attention
    2. Residual Connection
    3. Feed Forward Network
    4. Residual Connection
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
    ) -> None:
        super().__init__()

        self.self_attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
        )

        self.residual1 = ResidualConnection(
            d_model=d_model,
        )

        self.feed_forward = PositionwiseFeedForward(
            d_model=d_model,
            d_ff=d_ff,
        )

        self.residual2 = ResidualConnection(
            d_model=d_model,
        )

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:

        attention_output, _ = self.self_attention(
            query=x,
            key=x,
            value=x,
            mask=mask,
        )

        x = self.residual1(
            x=x,
            sublayer_output=attention_output,
        )

        feed_forward_output = self.feed_forward(x)

        x = self.residual2(
            x=x,
            sublayer_output=feed_forward_output,
        )

        return x