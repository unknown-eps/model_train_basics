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
        if d_model <= 0 or d_head <= 0 or d_model % d_head != 0:
            raise ValueError(
                "d_model and d_head must be positive, and d_head must divide d_model."
            )
        self.d_model = d_model
        self.d_head = d_head
        self.W_q, self.W_k, self.W_v = (
            nn.Linear(d_model, d_model),
            nn.Linear(d_model, d_model),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x):
        """Accept and return shape [B, S, D]: batch, sequence length, d_model."""
        if x.ndim != 3:
            raise ValueError(
                "Inputs to attention forward must have shape [B, S, D] (ndim=3)."
            )
        if x.shape[-1] != self.d_model:
            raise ValueError("Last dimension must match model dimension.")
        B, S, _ = x.shape
        d_model, d_head = self.d_model, self.d_head
        num_heads: int = d_model // d_head
        # Split each token's features into heads, then attend within each head.
        q = self.W_q(x).reshape(B, S, num_heads, d_head).transpose(1, 2)
        k = self.W_k(x).reshape(B, S, num_heads, d_head).transpose(1, 2)
        v = self.W_v(x).reshape(B, S, num_heads, d_head).transpose(1, 2)
        attn_matrices = (q @ k.transpose(-2, -1)) / d_head**0.5
        mask = torch.ones(S, S, device=x.device, dtype=torch.bool).triu(diagonal=1)
        attn_scores = attn_matrices.masked_fill(mask, float("-inf"))
        softmax_scores = torch.softmax(attn_scores, dim=-1)
        modified_reps = softmax_scores @ v  # B,num_heads,S,d_head
        # Group all heads for each token before merging their feature dimensions.
        return modified_reps.transpose(1, 2).reshape(B, S, d_model)


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
