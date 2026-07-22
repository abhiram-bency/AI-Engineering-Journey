"""
=========================================================
Transformer Configuration
=========================================================

Project:
    AI Engineering Journey

Description:
    Stores hyperparameters for the Transformer model.
"""

from dataclasses import dataclass


@dataclass
class TransformerConfig:
    """
    Configuration for the Transformer model.
    """

    # Vocabulary
    vocab_size: int = 10000

    # Maximum sequence length
    max_sequence_length: int = 512

    # Model dimensions
    d_model: int = 512
    d_ff: int = 2048

    # Attention
    num_heads: int = 8

    # Architecture
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6

    # Regularization
    dropout: float = 0.1