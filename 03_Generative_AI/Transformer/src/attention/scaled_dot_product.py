"""
=========================================================
Scaled Dot-Product Attention
=========================================================

Project:
    AI Engineering Journey

Module:
    Transformer from Scratch using PyTorch

Description:
    Implements the Scaled Dot-Product Attention mechanism
    introduced in the paper
    "Attention Is All You Need".

Reference:
    Vaswani et al., 2017
"""

import math

import torch
import torch.nn as nn
from torch import Tensor


class ScaledDotProductAttention(nn.Module):
    """
    Compute scaled dot-product attention.

    Input Shapes:
        Query:
            (..., seq_len_q, d_k)

        Key:
            (..., seq_len_k, d_k)

        Value:
            (..., seq_len_k, d_v)

    Returns:
        output

        attention_weights
    """

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:

        d_k = query.size(-1)

        attention_scores = (
            query
            @ key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:

            attention_scores = attention_scores.masked_fill(
                mask == 0,
                float("-inf"),
            )

        attention_weights = torch.softmax(
            attention_scores,
            dim=-1,
        )

        output = attention_weights @ value

        return output, attention_weights