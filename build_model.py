from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for the model.

    Supports vocab_size, num_layers, random_seed, d_model and d_head as input.
    For simplicity we assume only attention blocks, no ffn blocks.
    """

    vocab_size: int
    num_layers: int
    random_seed: int
    d_model: int
    d_head: int


if __name__ == "__main__":
    sample_config_dict = {
        "vocab_size": 10000,
        "num_layers": 4,
        "random_seed": 42,
        "d_model": 1024,
        "d_head": 64,
    }

    sample_model_config = ModelConfig(
        **sample_config_dict
    )
    print(sample_model_config)
