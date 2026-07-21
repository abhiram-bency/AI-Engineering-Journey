"""
=========================================================
Token Embedding Layer
=========================================================

Project:
    AI Engineering Journey

Module:
    Transformer from Scratch using PyTorch

Description:
    Implements the learnable token embedding layer used in
    the Transformer architecture proposed in
    "Attention Is All You Need".

    The embedding layer converts discrete token IDs into
    continuous dense vector representations.

Reference:
    Vaswani et al., 2017
    https://arxiv.org/abs/1706.03762
"""

from torch import Tensor
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    Learnable token embedding layer.

    Args:
        vocab_size (int):
            Number of tokens in the vocabulary.

        d_model (int):
            Embedding dimension.

    Input Shape:
        (batch_size, sequence_length)

    Output Shape:
        (batch_size, sequence_length, d_model)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
        )

    def forward(self, tokens: Tensor) -> Tensor:
        """
        Convert token IDs into embedding vectors.

        Args:
            tokens:
                Tensor containing token IDs.

        Returns:
            Embedded token representations.
        """
        return self.embedding(tokens)