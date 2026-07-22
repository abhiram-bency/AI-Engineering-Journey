# Transformer Encoder Layer

## Introduction

A Transformer Encoder Layer combines multiple reusable components into a single
building block.

## Architecture

Input

↓

Multi-Head Attention

↓

Residual Connection + LayerNorm

↓

Feed Forward Network

↓

Residual Connection + LayerNorm

↓

Output

## Why is it called Self-Attention?

In the encoder:

- Query = Input
- Key = Input
- Value = Input

Every token attends to every other token in the same sequence.

## Components Used

- Multi-Head Attention
- Position-wise Feed Forward Network
- Residual Connection
- Layer Normalization

## References

- Attention Is All You Need