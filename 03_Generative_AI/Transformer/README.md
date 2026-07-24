# Transformer from Scratch in PyTorch

> A clean, modular, and educational implementation of the original **Transformer architecture** proposed in **"Attention Is All You Need" (Vaswani et al., 2017)**, built entirely from first principles using PyTorch.

<p align="center">
  <img src="assets/images/transformer_architecture.png" width="800" alt="Transformer Architecture">
</p>

> **Note:** The architecture diagram will be added in a future update.

---

## Project Goals

This project is part of my **AI Engineering Journey**, where I re-implement fundamental machine learning and deep learning algorithms from scratch to understand how they work internally.

The objective is **not** to use high-level APIs such as:

- `torch.nn.Transformer`
- `torch.nn.MultiheadAttention`

Instead, every core component is implemented manually using basic PyTorch operations.

---

# Features

## Core Architecture

- ✅ Token Embedding
- ✅ Positional Encoding
- ✅ Scaled Dot-Product Attention
- ✅ Multi-Head Attention
- ✅ Position-wise Feed Forward Network
- ✅ Layer Normalization
- ✅ Residual Connections
- ✅ Encoder Layer
- ✅ Encoder
- ✅ Decoder Layer
- ✅ Decoder
- ✅ Complete Encoder-Decoder Transformer

---

## Utilities

- ✅ Attention Masks
- ✅ Xavier Weight Initialization
- ✅ Parameter Counter
- ✅ Reproducibility Utilities (Random Seeds)

---

## Software Engineering

- Modular project structure
- Unit tests
- Examples for every component
- Documentation for every module
- Beginner-friendly implementation

---

# Project Structure

```text
Transformer/

├── docs/
├── examples/
├── tests/
├── src/
│   ├── attention/
│   ├── layers/
│   ├── models/
│   └── utils/
│
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

# Components Implemented

| Component | Status |
|-----------|--------|
| Token Embedding | ✅ |
| Positional Encoding | ✅ |
| Scaled Dot Product Attention | ✅ |
| Multi Head Attention | ✅ |
| Feed Forward Network | ✅ |
| Layer Normalization | ✅ |
| Residual Connection | ✅ |
| Encoder Layer | ✅ |
| Encoder | ✅ |
| Decoder Layer | ✅ |
| Decoder | ✅ |
| Complete Transformer | ✅ |
| Attention Masks | ✅ |
| Xavier Initialization | ✅ |
| Parameter Counter | ✅ |
| Reproducibility Utilities | ✅ |

---

# Testing

The project includes unit tests for every implemented module.

Current status:

```text
16 tests passing
```

Run all tests:

```bash
pytest
```

---

# Examples

Each module includes a runnable demonstration inside the `examples/` directory.

Example:

```bash
python examples/transformer_demo.py
```

---

# Current Architecture

```text
Source Tokens
        │
Token Embedding
        │
Positional Encoding
        │
Encoder
        │
Encoder Output
        │
Target Tokens
        │
Token Embedding
        │
Positional Encoding
        │
Decoder
        │
Linear Projection
        │
Vocabulary Logits
```

---

# Roadmap

## Completed

- Transformer architecture
- Utility modules
- Unit testing
- Documentation

## Planned

- Dataset pipeline
- Tokenizer integration
- Training loop
- Label smoothing
- Noam learning rate scheduler
- Checkpoint management
- Greedy decoding
- Beam search
- End-to-end translation example
- Architecture visualizations
- GitHub Actions CI/CD

---

# References

- Vaswani et al., *Attention Is All You Need*, NeurIPS 2017.
- The Annotated Transformer
- PyTorch Documentation

---

# About

This repository is part of my **AI Engineering Journey**, where I implement AI and Machine Learning concepts from scratch to build a deeper understanding of modern AI systems.

More implementations covering Machine Learning, Deep Learning, Computer Vision, and Large Language Models will be added over time.