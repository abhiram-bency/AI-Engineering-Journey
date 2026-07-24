# Reproducibility

Machine learning experiments often rely on random initialization, data shuffling, and stochastic layers.

To ensure experiments can be reproduced, this project provides a utility function that sets the random seed for:

- Python's `random` module
- NumPy
- PyTorch (CPU)
- PyTorch (CUDA, if available)

It also configures cuDNN to use deterministic algorithms where possible.

## Why does this matter?

Reproducible experiments make debugging, benchmarking, and scientific comparisons much more reliable.