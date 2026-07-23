from src.config import TransformerConfig
from src.models.transformer import Transformer


def test_model_initialization():

    config = TransformerConfig()

    model = Transformer(config)

    assert model is not None