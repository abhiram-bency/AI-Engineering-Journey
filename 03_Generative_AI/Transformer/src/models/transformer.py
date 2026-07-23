"""
=========================================================
Complete Transformer
=========================================================

Project:
    AI Engineering Journey

Description:
    Full Transformer architecture composed of
    embeddings, positional encodings, encoder,
    decoder, and output projection.
"""

import torch.nn as nn
from torch import Tensor

from src.config import TransformerConfig
from src.layers.embedding import TokenEmbedding
from src.layers.positional_encoding import PositionalEncoding

from src.models.encoder import Encoder
from src.models.decoder import Decoder


class Transformer(nn.Module):

    def __init__(
        self,
        config: TransformerConfig,
    ) -> None:
        super().__init__()

        self.source_embedding = TokenEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
        )

        self.target_embedding = TokenEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
        )

        self.positional_encoding = PositionalEncoding(
            d_model=config.d_model,
            max_sequence_length=config.max_sequence_length,
        )

        self.encoder = Encoder(config)

        self.decoder = Decoder(config)

        self.output_projection = nn.Linear(
            config.d_model,
            config.vocab_size,
        )

    def forward(
        self,
        source_tokens: Tensor,
        target_tokens: Tensor,
        source_mask=None,
        target_mask=None,
    ):

        source = self.source_embedding(source_tokens)
        source = self.positional_encoding(source)

        encoder_output = self.encoder(
            source,
            mask=source_mask,
        )

        target = self.target_embedding(target_tokens)
        target = self.positional_encoding(target)

        decoder_output = self.decoder(
            x=target,
            encoder_output=encoder_output,
            target_mask=target_mask,
            source_mask=source_mask,
        )

        logits = self.output_projection(
            decoder_output
        )

        return logits