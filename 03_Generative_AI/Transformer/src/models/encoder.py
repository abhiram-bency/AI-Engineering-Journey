"""
=========================================================
Transformer Encoder
=========================================================
"""

import torch.nn as nn
from torch import Tensor

from src.config import TransformerConfig
from src.models.encoder_layer import EncoderLayer


class Encoder(nn.Module):
    """
    Transformer Encoder consisting of multiple Encoder Layers.
    """

    def __init__(
        self,
        config: TransformerConfig,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    d_ff=config.d_ff,
                )
                for _ in range(config.num_encoder_layers)
            ]
        )

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:

        for layer in self.layers:
            x = layer(
                x=x,
                mask=mask,
            )

        return x