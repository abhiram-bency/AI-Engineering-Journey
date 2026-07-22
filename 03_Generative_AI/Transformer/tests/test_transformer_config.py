from src.config import TransformerConfig


def test_default_config():

    config = TransformerConfig()

    assert config.d_model == 512
    assert config.num_heads == 8
    assert config.d_ff == 2048