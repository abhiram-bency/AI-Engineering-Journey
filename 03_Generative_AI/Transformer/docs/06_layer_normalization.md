## Why Layer Normalization?

Layer Normalization normalizes the features of each token independently.

Unlike Batch Normalization, it does not depend on other samples in the batch.

This makes it ideal for sequence models such as Transformers, where batch size
and sequence length may vary.

### Mathematical Formulation

Given an input vector:

\[
x = [x_1, x_2, ..., x_d]
\]

Compute the mean:

\[
\mu = \frac{1}{d}\sum_{i=1}^{d}x_i
\]

Compute the variance:

\[
\sigma^2 = \frac{1}{d}\sum_{i=1}^{d}(x_i-\mu)^2
\]

Normalize:

\[
\hat{x}=\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}
\]

Finally,

\[
y=\gamma\hat{x}+\beta
\]