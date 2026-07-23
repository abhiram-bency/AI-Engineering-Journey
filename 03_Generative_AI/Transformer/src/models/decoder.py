"""
=========================================================
Transformer Decoder
=========================================================

Project:
    AI Engineering Journey

Description:
    Stacks multiple Transformer Decoder Layers.
"""

import torch.nn as nn
from torch import Tensor

from src.config import TransformerConfig
from src.models.decoder_layer import DecoderLayer


class Decoder(nn.Module):
    """
    Transformer Decoder consisting of multiple Decoder Layers.
    """

    def __init__(
        self,
        config: TransformerConfig,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    d_ff=config.d_ff,
                )
                for _ in range(config.num_decoder_layers)
            ]
        )

    def forward(
        self,
        x: Tensor,
        encoder_output: Tensor,
        target_mask: Tensor | None = None,
        source_mask: Tensor | None = None,
    ) -> Tensor:

        for layer in self.layers:
            x = layer(
                x=x,
                encoder_output=encoder_output,
                target_mask=target_mask,
                source_mask=source_mask,
            )

        return x