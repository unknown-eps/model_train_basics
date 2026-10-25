from copy import deepcopy

import pytest
import torch

from model import attention_module, flash_attention_module


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Flash Attention requires a CUDA GPU."
)


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
@pytest.mark.parametrize("d_model,d_head", [(16, 8), (512, 256)])
def test_flash_attention_matches_float32_reference(compiled, d_model, d_head):
    torch.manual_seed(42)
    source = (
        attention_module(d_model, d_head).to(device="cuda", dtype=torch.bfloat16).eval()
    )
    reference = deepcopy(source).float()
    flash = flash_attention_module.from_attention_module(source)
    if compiled:
        flash = torch.compile(flash, backend="inductor", dynamic=False)
    x = torch.randn(2, 7, d_model, device="cuda", dtype=torch.bfloat16)

    with torch.inference_mode():
        expected = reference(x.float())
        actual = flash(x)

    assert actual.shape == x.shape
    assert actual.dtype == torch.bfloat16
    # BF16 has fewer mantissa bits; compare against FP32 with BF16 tolerances.
    torch.testing.assert_close(actual.float(), expected, rtol=2e-2, atol=5e-3)


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
def test_flash_attention_is_causal(compiled):
    torch.manual_seed(42)
    source = attention_module(512, 256).to(device="cuda", dtype=torch.bfloat16).eval()
    flash = flash_attention_module.from_attention_module(source)
    if compiled:
        flash = torch.compile(flash, backend="inductor", dynamic=False)
    x = torch.randn(2, 7, 512, device="cuda", dtype=torch.bfloat16)
    changed = x.clone()
    changed[:, 4:] = torch.randn_like(changed[:, 4:])

    with torch.inference_mode():
        original = flash(x)
        modified = flash(changed)

    torch.testing.assert_close(original[:, :4], modified[:, :4], rtol=0, atol=0)
