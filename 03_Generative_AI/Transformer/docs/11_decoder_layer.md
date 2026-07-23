# Transformer Decoder Layer

## Overview

The Decoder Layer is responsible for generating output representations while attending to both:

1. Previously generated target tokens (Masked Self-Attention)
2. Encoder outputs (Cross-Attention)

It consists of three major sublayers:

- Masked Multi-Head Self-Attention
- Encoder-Decoder Cross-Attention
- Position-wise Feed Forward Network

Each sublayer is followed by a Residual Connection and Layer Normalization.

## Architecture

Input
│
├── Masked Self-Attention
│
├── Add & LayerNorm
│
├── Cross-Attention
│
├── Add & LayerNorm
│
├── Feed Forward
│
├── Add & LayerNorm
│
▼
Output

## Self-Attention vs Cross-Attention

### Self-Attention

Q = K = V = Decoder Input

### Cross-Attention

Q = Decoder Output

K = Encoder Output

V = Encoder Output