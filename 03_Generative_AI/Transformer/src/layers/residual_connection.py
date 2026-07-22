"""
=========================================================
Residual Connection
=========================================================

Project:
    AI Engineering Journey

Module:
    Transformer from Scratch using PyTorch

Description:
    Implements the residual (skip) connection used in the
    original Transformer architecture.

Reference:
    Vaswani et al., 2017
"""

import torch.nn as nn
from torch import Tensor

from .layer_normalization import LayerNormalization


class ResidualConnection(nn.Module):
    """
    Residual Connection followed by Layer Normalization
    (Post-LayerNorm as used in the original Transformer).
    """

    def __init__(
        self,
        d_model: int,
    ) -> None:
        super().__init__()

        self.layer_norm = LayerNormalization(d_model)

    def forward(
        self,
        x: Tensor,
        sublayer_output: Tensor,
    ) -> Tensor:
        """
        Args:
            x:
                Original input tensor.

            sublayer_output:
                Output from a sublayer such as Multi-Head Attention
                or Feed Forward Network.

        Returns:
            Tensor with residual connection applied.
        """

        output = x + sublayer_output
        output = self.layer_norm(output)

        return output