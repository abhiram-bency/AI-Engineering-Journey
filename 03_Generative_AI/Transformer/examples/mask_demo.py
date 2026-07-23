import torch

from src.utils.mask import (
    create_padding_mask,
    create_look_ahead_mask,
)


sequence = torch.tensor([
    [5, 8, 2, 0, 0]
])

padding = create_padding_mask(sequence)

look_ahead = create_look_ahead_mask(5)

print("Padding Mask")
print(padding)

print()

print("Look Ahead Mask")
print(look_ahead)