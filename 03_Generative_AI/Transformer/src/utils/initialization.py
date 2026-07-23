"""
=========================================================
Weight Initialization Utilities
=========================================================

Provides Xavier initialization for Transformer models.
"""

import torch.nn as nn


def initialize_transformer(model: nn.Module) -> None:
    """
    Initialize all linear layer weights using
    Xavier Uniform initialization.
    """

    for parameter in model.parameters():

        if parameter.dim() > 1:
            nn.init.xavier_uniform_(parameter)