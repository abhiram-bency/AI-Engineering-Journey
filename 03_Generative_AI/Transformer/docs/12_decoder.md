# Transformer Decoder

## Overview

The Transformer Decoder consists of multiple identical Decoder Layers stacked sequentially.

Each Decoder Layer contains:

- Masked Multi-Head Self-Attention
- Cross-Attention
- Position-wise Feed Forward Network
- Residual Connections
- Layer Normalization

The decoder generates contextual representations while attending to both previously generated tokens and the encoder output.

## Architecture

Target Embedding

↓

Positional Encoding

↓

Decoder Layer × N

↓

Decoder Output