# Multi-Head Attention

## Introduction

Multi-Head Attention extends Scaled Dot-Product Attention by allowing the model
to attend to different representation subspaces simultaneously.

Instead of performing a single attention operation, the input is projected into
multiple attention heads, each learning different relationships between tokens.

---

## Forward Pass

The forward pass consists of five stages:

1. Project the input into Query, Key, and Value representations.
2. Split each projection into multiple attention heads.
3. Compute Scaled Dot-Product Attention independently for every head.
4. Combine the outputs of all attention heads.
5. Apply a final linear projection to produce the output.

This modular implementation closely follows the architecture described in
*Attention Is All You Need* while keeping each responsibility isolated in a
dedicated helper method.


## Complexity Analysis

Let

- n = sequence length
- d = embedding dimension
- h = number of heads

Time Complexity

O(n² · d)

Space Complexity

O(n²)

The quadratic dependence on sequence length comes from the attention matrix.

This is the primary computational bottleneck of the original Transformer.

## Why Multiple Heads?

---

## Architecture

---

## Mathematical Formulation

---

## Tensor Shape Transformation

Input

(batch_size, sequence_length, d_model)

↓

Split

(batch_size, num_heads, sequence_length, head_dimension)

↓

Scaled Dot-Product Attention

↓

Combine

(batch_size, sequence_length, d_model)

---

## Splitting Heads

---

## Combining Heads

---

## Why transpose()?

---

## Why contiguous()?

---

## Common Mistakes

---

## Interview Questions

---

## References

1. Vaswani et al.
   Attention Is All You Need
   NeurIPS 2017

2. The Annotated Transformer

3. PyTorch Documentation