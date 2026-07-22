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

    def _split_heads(
        self,
        tensor: Tensor,
    ) -> Tensor:
        """
        Split the embedding dimension into multiple attention heads.

        Args:
            tensor:
                Shape:
                (batch_size, sequence_length, d_model)

        Returns:
            Tensor of shape:
            (batch_size, num_heads, sequence_length, head_dimension)
        """

        batch_size, sequence_length, _ = tensor.size()

        # (batch_size, sequence_length, d_model)
        # ->
        # (batch_size, sequence_length, num_heads, head_dimension)
        tensor = tensor.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dimension,
        )

        # (batch_size, sequence_length, num_heads, head_dimension)
        # ->
        # (batch_size, num_heads, sequence_length, head_dimension)
        tensor = tensor.transpose(1, 2)

        return tensor
    
    def _combine_heads(
        self,
        tensor: Tensor,
    ) -> Tensor:
        """
        Combine multiple attention heads into a single tensor.

        Args:
            tensor:
                Shape:
                (batch_size, num_heads, sequence_length, head_dimension)

        Returns:
            Tensor of shape:
            (batch_size, sequence_length, d_model)
        """

        batch_size, _, sequence_length, _ = tensor.size()

        # (batch_size, num_heads, sequence_length, head_dimension)
        # ->
        # (batch_size, sequence_length, num_heads, head_dimension)
        tensor = tensor.transpose(1, 2)

        # Ensure the tensor occupies contiguous memory before reshaping.
        tensor = tensor.contiguous()

        # (batch_size, sequence_length, num_heads, head_dimension)
        # ->
        # (batch_size, sequence_length, d_model)
        tensor = tensor.view(
            batch_size,
            sequence_length,
            self.d_model,
        )

        return tensor
    
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Compute Multi-Head Attention.

        Args:
            query:
                Shape:
                (batch_size, sequence_length, d_model)

            key:
                Shape:
                (batch_size, sequence_length, d_model)

            value:
                Shape:
                (batch_size, sequence_length, d_model)

            mask:
                Optional attention mask.

        Returns:
            output:
                (batch_size, sequence_length, d_model)

            attention_weights:
                (batch_size, num_heads, sequence_length, sequence_length)
        """

        # Step 1: Linear projections
        query = self.query_projection(query)
        key = self.key_projection(key)
        value = self.value_projection(value)

        # Step 2: Split into multiple heads
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        # Step 3: Scaled Dot-Product Attention
        output, attention_weights = self.attention(
            query=query,
            key=key,
            value=value,
            mask=mask,
        )

        # Step 4: Combine attention heads
        output = self._combine_heads(output)

        # Step 5: Final linear projection
        output = self.output_projection(output)

        return output, attention_weights