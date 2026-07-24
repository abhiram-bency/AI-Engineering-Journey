from src.config import TransformerConfig
from src.models.transformer import Transformer

from src.utils.parameter_count import (
    count_parameters,
    count_all_parameters,
)


def test_parameter_counter():

    config = TransformerConfig()

    model = Transformer(config)

    trainable = count_parameters(model)

    total = count_all_parameters(model)

    assert trainable > 0

    assert total >= trainable