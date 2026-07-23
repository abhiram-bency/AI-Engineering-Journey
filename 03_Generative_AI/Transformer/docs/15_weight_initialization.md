# Weight Initialization

Neural networks require proper weight initialization to train effectively.

This project uses **Xavier Uniform Initialization**, introduced by Glorot & Bengio (2010).

## Why?

Good initialization helps:

- stabilize gradients
- speed up convergence
- avoid exploding activations
- avoid vanishing gradients

Every learnable matrix in the Transformer is initialized using Xavier Uniform.