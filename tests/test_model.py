from pathlib import Path

import pytest
import torch
import yaml
from pydantic import ValidationError

from build_model import ModelConfig
from model import attention_module, builtin_attention_module


def test_config_rejects_extra_field(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "vocab_size": 100,
                "num_layers": 2,
                "random_seed": 42,
                "d_model": 12,
                "d_head": 6,
                "unexpected_field": True,
            }
        )
    )

    with pytest.raises(ValidationError) as exc_info:
        ModelConfig.from_yaml(config_path)

    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["type"] == "extra_forbidden"
    assert errors[0]["loc"] == ("unexpected_field",)


def test_attention_matches_loop_reference():
    torch.manual_seed(42)
    B, S, D = 2, 3, 12
    d_head = 6
    module = attention_module(D, d_head).cpu()
    x = torch.randn(B, S, D, device="cpu")

    with torch.no_grad():
        actual = module(x)
        # Share the learned projections, but compute attention without head reshapes
        # or batched matrix multiplication so this is an independent reference.
        q, k, v = module.W_q(x), module.W_k(x), module.W_v(x)
        expected = torch.zeros_like(x)
        for batch in range(B):
            for head in range(D // d_head):
                features = slice(head * d_head, (head + 1) * d_head)
                for query in range(S):
                    # Only current/past keys participate: future weights are zero.
                    scores = torch.stack(
                        [
                            torch.dot(q[batch, query, features], k[batch, key, features])
                            / d_head**0.5
                            for key in range(query + 1)
                        ]
                    )
                    weights = torch.softmax(scores, dim=0)
                    for key in range(query + 1):
                        expected[batch, query, features] += (
                            weights[key] * v[batch, key, features]
                        )
        expected = module.W_o(expected)

    assert actual.shape == (B, S, D)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("use_builtin", [False, True], ids=["custom", "builtin"])
def test_compiled_attention_matches_eager(use_builtin):
    torch.manual_seed(42)
    custom = attention_module(d_model=12, d_head=6).cpu().eval()
    module = builtin_attention_module.from_attention_module(custom) if use_builtin else custom
    compiled = torch.compile(module, backend="inductor", dynamic=False)
    x = torch.randn(2, 3, 12, device="cpu")

    with torch.inference_mode():
        expected = custom(x)
        actual = compiled(x)

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_attention_matches_builtin():
    torch.manual_seed(42)
    module = attention_module(d_model=12, d_head=6).cpu()
    builtin = builtin_attention_module.from_attention_module(module)
    x = torch.randn(2, 3, 12, device="cpu")

    with torch.no_grad():
        expected = module(x)
        actual = builtin(x)

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
