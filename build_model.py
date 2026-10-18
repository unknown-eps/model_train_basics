from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, field_validator
from pydantic.types import PositiveInt


class ModelConfig(BaseModel):
    """Configuration for the model.

    Supports vocab_size, num_layers, random_seed, d_model and d_head as input.
    For simplicity we assume only attention blocks, no ffn blocks.
    """

    model_config = ConfigDict(extra="forbid")

    vocab_size: PositiveInt
    num_layers: PositiveInt
    random_seed: PositiveInt
    d_model: PositiveInt
    d_head: PositiveInt

    @field_validator("d_head")
    @classmethod
    def d_head_divides_d_model(cls, d_head: int, info) -> int:
        d_model = info.data.get("d_model")
        if d_model is not None and d_model % d_head != 0:
            raise ValueError(f"d_head ({d_head}) must divide d_model ({d_model})")
        return d_head

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelConfig":
        """Load and validate a config from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)


if __name__ == "__main__":
    sample_config_dict = {
        "vocab_size": 10000,
        "num_layers": 4,
        "random_seed": 42,
        "d_model": 1024,
        "d_head": 64,
    }

    sample_model_config = ModelConfig(**sample_config_dict)
    print(sample_model_config)

    loaded_config = ModelConfig.from_yaml("sample_configs/sample_config.yaml")
    print(loaded_config)
