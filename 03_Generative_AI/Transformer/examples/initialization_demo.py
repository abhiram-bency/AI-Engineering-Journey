from src.config import TransformerConfig
from src.models.transformer import Transformer


config = TransformerConfig()

model = Transformer(config)

print("Model successfully initialized.")

first_parameter = next(model.parameters())

print(first_parameter.shape)
print(first_parameter.mean().item())
print(first_parameter.std().item())