import torch

from src.utils.seed import set_seed


def test_seed_reproducibility():

    set_seed(42)

    tensor1 = torch.randn(3, 3)

    set_seed(42)

    tensor2 = torch.randn(3, 3)

    assert torch.equal(tensor1, tensor2)