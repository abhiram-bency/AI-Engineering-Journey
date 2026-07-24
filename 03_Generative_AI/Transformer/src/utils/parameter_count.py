"""
=========================================================
Model Parameter Utilities
=========================================================

Provides helper functions for counting
trainable and non-trainable parameters.
"""

import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    """
    Returns the number of trainable parameters.
    """

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def count_all_parameters(model: nn.Module) -> int:
    """
    Returns the total number of parameters.
    """

    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )