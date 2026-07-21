"""
=========================================================
Positional Encoding
=========================================================

Project:
    AI Engineering Journey

Module:
    Transformer from Scratch using PyTorch

Description:
    Implements the sinusoidal positional encoding proposed
    in the original Transformer paper.

    Since the Transformer contains no recurrence or
    convolution, positional information is added to the
    token embeddings using deterministic sine and cosine
    functions.

Paper:
    Vaswani et al., 2017
    https://arxiv.org/abs/1706.03762
"""

import math

import torch
import torch.nn as nn
from torch import Tensor


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.

    Args:
        d_model:
            Embedding dimension.

        max_sequence_length:
            Maximum supported sequence length.

    Input Shape:
        (batch_size, sequence_length, d_model)

    Output Shape:
        (batch_size, sequence_length, d_model)
    """

    def __init__(
        self,
        d_model: int,
        max_sequence_length: int = 5000,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.max_sequence_length = max_sequence_length

        positional_encoding = self._create_positional_encoding()

        self.register_buffer(
            "positional_encoding",
            positional_encoding,
        )

    def _create_positional_encoding(self) -> Tensor:
        """
        Create the sinusoidal positional encoding matrix.

        Returns:
            Tensor of shape
            (1, max_sequence_length, d_model)
        """

        position = torch.arange(
            self.max_sequence_length,
            dtype=torch.float32,
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                self.d_model,
                2,
                dtype=torch.float32,
            )
            * (-math.log(10000.0) / self.d_model)
        )

        positional_encoding = torch.zeros(
            self.max_sequence_length,
            self.d_model,
        )

        positional_encoding[:, 0::2] = torch.sin(position * div_term)

        positional_encoding[:, 1::2] = torch.cos(position * div_term)

        return positional_encoding.unsqueeze(0)

    def forward(
        self,
        embeddings: Tensor,
    ) -> Tensor:
        """
        Add positional encoding to token embeddings.

        Args:
            embeddings:
                Shape:
                (batch_size, sequence_length, d_model)

        Returns:
            Tensor with positional information added.
        """

        sequence_length = embeddings.size(1)

        return (
            embeddings
            + self.positional_encoding[:, :sequence_length]
        )