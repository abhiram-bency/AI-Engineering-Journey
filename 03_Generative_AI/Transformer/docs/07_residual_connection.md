# Residual Connection

## Introduction

Residual (skip) connections allow information to flow directly from the input
to the output of a sublayer.

Instead of learning a complete transformation, the model learns a residual
correction to the input.

## Mathematical Formulation

Given an input tensor `x` and a sublayer output `F(x)`:

\[
y = \text{LayerNorm}(x + F(x))
\]

This is the Post-LayerNorm formulation used in the original Transformer paper.

## Why Residual Connections?

- Improve gradient flow.
- Enable deeper networks.
- Preserve original token representations.
- Stabilize training.

## Modern LLM Note

- Transformer (2017): Post-LayerNorm
- GPT-2 / GPT-3: Pre-LayerNorm
- LLaMA / Gemma / Mistral: Pre-LayerNorm + RMSNorm