# Transformer Configuration

## Why a Configuration Class?

As the project grows, many components require the same hyperparameters:

- d_model
- d_ff
- num_heads
- num_encoder_layers
- num_decoder_layers

Passing these individually makes constructors verbose and harder to maintain.

A configuration object centralizes these values, making the code cleaner and more extensible.

## Benefits

- Single source of truth
- Easier experimentation
- Cleaner constructors
- Production-style design