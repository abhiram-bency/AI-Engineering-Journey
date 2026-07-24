import torch

from src.utils.seed import set_seed

set_seed(42)

tensor1 = torch.randn(2, 2)

set_seed(42)

tensor2 = torch.randn(2, 2)

print("Tensor 1")
print(tensor1)

print()

print("Tensor 2")
print(tensor2)

print()

print("Are Equal:", torch.equal(tensor1, tensor2))