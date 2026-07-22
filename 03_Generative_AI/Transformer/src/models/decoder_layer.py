"""
=========================================================
Transformer Decoder Layer
=========================================================

Project:
    AI Engineering Journey

Module:
    Transformer from Scratch using PyTorch

Description:
    Implements a single Transformer Decoder Layer as
    described in "Attention Is All You Need".

Reference:
    Vaswani et al., 2017
"""

import torch.nn as nn
from torch import Tensor

from src.attention.multi_head_attention import MultiHeadAttention
from src.layers.feed_forward import PositionwiseFeedForward
from src.layers.residual_connection import ResidualConnection


class DecoderLayer(nn.Module):
    """
    Single Transformer Decoder Layer.

    Consists of:

    1. Masked Self-Attention
    2. Residual Connection
    3. Cross Attention
    4. Residual Connection
    5. Feed Forward Network
    6. Residual Connection
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
    ) -> None:
        super().__init__()

        # Masked Self-Attention
        self.self_attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
        )

        # Cross Attention
        self.cross_attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
        )

        # Feed Forward Network
        self.feed_forward = PositionwiseFeedForward(
            d_model=d_model,
            d_ff=d_ff,
        )

        # Residual Connections
        self.residual1 = ResidualConnection(d_model)
        self.residual2 = ResidualConnection(d_model)
        self.residual3 = ResidualConnection(d_model)

    def forward(
        self,
        x: Tensor,
        encoder_output: Tensor,
        target_mask: Tensor | None = None,
        source_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Forward pass of a Transformer Decoder Layer.

        Args:
            x:
                Decoder input.

            encoder_output:
                Output from the encoder.

            target_mask:
                Causal mask for decoder self-attention.

            source_mask:
                Optional mask for encoder-decoder attention.

        Returns:
            Decoder output tensor.
        """

        # -----------------------------
        # 1. Masked Self-Attention
        # -----------------------------
        self_attention_output, _ = self.self_attention(
            query=x,
            key=x,
            value=x,
            mask=target_mask,
        )

        x = self.residual1(
            x=x,
            sublayer_output=self_attention_output,
        )

        # -----------------------------
        # 2. Cross Attention
        # -----------------------------
        cross_attention_output, _ = self.cross_attention(
            query=x,
            key=encoder_output,
            value=encoder_output,
            mask=source_mask,
        )

        x = self.residual2(
            x=x,
            sublayer_output=cross_attention_output,
        )

        # -----------------------------
        # 3. Feed Forward Network
        # -----------------------------
        feed_forward_output = self.feed_forward(x)

        x = self.residual3(
            x=x,
            sublayer_output=feed_forward_output,
        )

        return x