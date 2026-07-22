# Transformer Encoder

## Introduction

The Transformer Encoder is composed of multiple identical Encoder Layers stacked sequentially.

Each Encoder Layer contains:

- Multi-Head Self-Attention
- Residual Connection + Layer Normalization
- Position-wise Feed Forward Network
- Residual Connection + Layer Normalization

## Why Stack Multiple Layers?

Each layer refines the token representations by incorporating increasingly complex contextual information.

The original Transformer Base model uses 6 encoder layers.

## Implementation Notes

This implementation uses `torch.nn.ModuleList` to register and manage the stack of encoder layers.