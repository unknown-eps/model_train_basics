"""Command-line setup for the attention-only model."""

import argparse
from pathlib import Path

import torch
import yaml
from torch import nn

from build_model import ModelConfig


def parsing_arguments() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config_path",
        type=Path,
        required=True,
        help="Path to the model's YAML configuration file.",
    )
    return parser.parse_args()


class attention_module(nn.Module):
    """Defines the attention module"""

    def __init__(self, d_model, d_head):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.W_q, self.W_k, self.W_v = (
            nn.Linear(d_model, d_model),
            nn.Linear(d_model, d_model),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x):
        """X can be shape B,S,D or S,D where B,S,D denote batch,seq len and d_model respectively"""
        if x.ndim not in (2,3):
            raise ValueError(
                "Inputs to attention forward should have shape ndim 2 or 3"
            )
        if x.ndim == 2:
            x = x.reshape(-1, *x.shape[0])
        # x is B,S,D at this point
        if x.shape[-1]!=self.d_model:
            raise ValueError("Last dimension must match model dimension.")


class llama_style_model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size, embedding_dim=config.d_model
        )
        self.norm = nn.RMSNorm(normalized_shape=config.d_model, eps=1e-5)


def main() -> None:
    args = parsing_arguments()

    try:
        config = ModelConfig.from_yaml(args.config_path)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise ValueError(
            f"Could not load configuration from {args.config_path}"
        ) from exc

    # Seed before creating layers so their initial weights are repeatable.
    # manual_seed seeds PyTorch on both CPU and CUDA devices.
    torch.manual_seed(config.random_seed)

    print(config)


if __name__ == "__main__":
    main()
