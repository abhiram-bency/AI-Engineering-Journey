# Attention Masks

Transformer models require masking to prevent attention to invalid positions.

## Padding Mask

Masks PAD tokens so that attention ignores them.

Example:

Input:

```
[5, 8, 2, 0, 0]
```

Mask:

```
T T T F F
```

---

## Look Ahead Mask

Prevents decoder positions from attending to future tokens.

Example:

```
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

This enables autoregressive generation.