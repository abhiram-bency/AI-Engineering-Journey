"""
=========================================================
Layer Normalization
=========================================================

Project:
    AI Engineering Journey

Module:
    Transformer from Scratch using PyTorch

Description:
    Implements Layer Normalization from first principles,
    following the equations described in the original paper.

Reference:
    Ba, Kiros, Hinton (2016)
    Layer Normalization
"""

import torch
import torch.nn as nn
from torch import Tensor


class LayerNormalization(nn.Module):
    """
    Layer Normalization.

    Args:
        d_model:
            Embedding dimension.

        epsilon:
            Small constant added for numerical stability.
    """

    def __init__(
        self,
        d_model: int,
        epsilon: float = 1e-5,
    ) -> None:
        super().__init__()

        self.epsilon = epsilon

        # Learnable scaling parameter (γ)
        self.gamma = nn.Parameter(torch.ones(d_model))

        # Learnable shifting parameter (β)
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Args:
            x:
                Shape:
                (batch_size, sequence_length, d_model)

        Returns:
            Tensor with the same shape.
        """

        # Compute mean across the embedding dimension
        mean = x.mean(
            dim=-1,
            keepdim=True,
        )

        # Compute variance across the embedding dimension
        variance = x.var(
            dim=-1,
            keepdim=True,
            unbiased=False,
        )

        # Normalize
        normalized = (
            x - mean
        ) / torch.sqrt(
            variance + self.epsilon
        )

        # Scale and shift
        output = (
            self.gamma * normalized
            + self.beta
        )

        return output