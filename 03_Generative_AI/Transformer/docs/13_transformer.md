# Complete Transformer

## Overview

This module combines all previously implemented components into the complete Transformer architecture proposed in the paper **Attention Is All You Need**.

Pipeline:

Source Tokens
↓
Embedding
↓
Positional Encoding
↓
Encoder
↓
Encoder Output
↓
Decoder
↓
Linear Projection
↓
Vocabulary Logits

## Components

- Token Embedding
- Positional Encoding
- Encoder
- Decoder
- Output Projection Layer

The output projection converts decoder representations into vocabulary logits for next-token prediction.