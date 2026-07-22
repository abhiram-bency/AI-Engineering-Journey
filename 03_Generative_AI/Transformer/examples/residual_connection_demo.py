"""
Residual Connection Demo

This script demonstrates how a residual connection combines the
original input with the output of a sublayer, followed by
Layer Normalization.
"""

import torch

from src.layers.residual_connection import ResidualConnection


def main():

    batch_size = 2
    sequence_length = 5
    d_model = 512

    residual = ResidualConnection(d_model=d_model)

    # Original input
    x = torch.randn(
        batch_size,
        sequence_length,
        d_model,
    )

    # Simulated output from a sublayer
    sublayer_output = torch.randn(
        batch_size,
        sequence_length,
        d_model,
    )

    output = residual(
        x=x,
        sublayer_output=sublayer_output,
    )

    print("Input Shape           :", x.shape)
    print("Sublayer Output Shape :", sublayer_output.shape)
    print("Final Output Shape    :", output.shape)


if __name__ == "__main__":
    main()