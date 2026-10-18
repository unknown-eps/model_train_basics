"""Command-line setup for the attention-only model."""

import argparse
from pathlib import Path

import torch
import yaml

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
