"""
=========================================================
Position-wise Feed Forward Network
=========================================================

Project:
    AI Engineering Journey

Module:
    Transformer from Scratch using PyTorch

Description:
    Implements the position-wise feed forward network
    introduced in "Attention Is All You Need".

Reference:
    Vaswani et al., 2017
"""

import torch.nn as nn
from torch import Tensor


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed Forward Network.

    Args:
        d_model:
            Embedding dimension.

        d_ff:
            Hidden dimension of the feed forward network.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
    ) -> None:
        super().__init__()

        self.linear1 = nn.Linear(
            d_model,
            d_ff,
        )

        self.activation = nn.ReLU()

        self.linear2 = nn.Linear(
            d_ff,
            d_model,
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Args:
            x:
                (batch_size, sequence_length, d_model)

        Returns:
            (batch_size, sequence_length, d_model)
        """

        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)

        return x