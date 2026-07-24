from src.config import TransformerConfig
from src.models.transformer import Transformer

from src.utils.parameter_count import (
    count_parameters,
    count_all_parameters,
)

config = TransformerConfig()

model = Transformer(config)

print("=" * 50)
print("Transformer Model Statistics")
print("=" * 50)

print(f"Trainable Parameters : {count_parameters(model):,}")
print(f"Total Parameters     : {count_all_parameters(model):,}")