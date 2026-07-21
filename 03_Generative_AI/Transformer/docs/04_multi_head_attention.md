# Multi-Head Attention

## Introduction

Multi-Head Attention extends Scaled Dot-Product Attention by allowing the model
to attend to different representation subspaces simultaneously.

Instead of performing a single attention operation, the input is projected into
multiple attention heads, each learning different relationships between tokens.

---

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