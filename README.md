# Objectives
1. To build a small llama-like model from the best principles.
2. Integrade it with HF transformers library.
3. Enable Flash Attention 2 and profile it with standard attention.
4. Add pre-commit hooks to format and run some tests.
5. Add cpu based tests so that it can run on github ci.

## Framework
1. Pixi for environment setup.
2. We will setup two groups first for gpu and then for cpu only ci.
3. We will use pytest for testing and ruff for linting and formatting.
4. ipykerel will be installed in the gpu env for quick testing with vscode.

## Steps
1. Setting up torch, transformers and ipykernel for gpu group 
2. Write the main code.
3. Write flash attention support and profile time and memory
4. Write tests
5. Write linting support and auto-format on commits.
6. Setting up github ci.

## Run the model setup

```bash
pixi run -e gpu python model.py --config_path sample_configs/sample_config.yaml
```

`--config_path` is required. The entry point loads and validates the YAML using
`ModelConfig` from `build_model.py`, seeds PyTorch before any layers are created,
and prints the configuration. Model layers are not implemented yet.

## Profile attention

```bash
pixi run -e gpu python profile_attention.py
```

Profiles eager and compiled versions of custom, built-in, and Flash Attention separately
at sequence lengths 1024 and 2048, with batch size 4, model dimension 2048, and 8
heads. All variants use BF16. Flash uses PyTorch SDPA with only
`SDPBackend.FLASH_ATTENTION` enabled, so unsupported inputs fail instead of silently
falling back. No separate `flash-attn` package is needed. Compilation uses
TorchInductor with `dynamic=False` and default mode.
Each variant gets 25 warmup forwards before one recorded inference forward;
compilation is excluded from the capture. Output checks live only in pytest.
Each variant and sequence length runs in a separate process, so eager captures
do not contain compiler workers left over from other variants. Python stack
recording is on by default to show source call information. Add `--no-with_stack`
for a cleaner timeline. Shapes, CUDA activity, and memory recording remain on.
To capture just one case, use `--variant builtin --seq_len 2048` (or select
`custom`, `custom_compiled`, `builtin_compiled`, `flash`, or `flash_compiled`).

The ignored `profiles/` directory contains twelve `.txt` timing/allocation reports
and twelve `.json` CUDA traces after a complete run. Open a JSON trace at <https://ui.perfetto.dev> to
inspect operators and their CUDA kernels. Use `--output_dir PATH` to change the
destination. Profiling adds overhead; use `benchmark.py` for performance numbers.
Allocation tables show net operator allocations, not peak live memory.

Earlier float32 profiles showed that the eager built-in uses native multi-head attention,
not Flash Attention, for both sizes. Q/K/V share one projection. At sequence
length 1024, masked softmax directly launches its kernel; at 2048 it instead
clones the score tensor, applies the mask, and runs a separate softmax. This
adds large temporary allocations and helps explain the benchmark's memory jump.
BF16 Flash profiles show `aten::_scaled_dot_product_flash_attention` and a
`pytorch_flash::flash_fwd_kernel` for both eager and compiled Flash variants.

## Benchmark attention

```bash
pixi run -e gpu python benchmark.py
```

Compares the same six BF16 variants at sequence lengths 128 through 2048, using 25
warmups and 100 measured forwards per variant. Compilation happens during warmup,
outside timing. Reports mean CUDA-event latency, speedup relative to eager custom
attention, and peak extra allocated memory per forward. All variants use matching
weights; compiled wrappers share their eager module's parameters.
Saves `profiles/benchmark.png` with time and mean per-forward peak memory bars,
including ±1 sample standard deviation. Memory uses 100 separate forward runs.

Sequence length is capped at 2048 because the eager built-in implementation ran
out of memory at 4096 on this GPU.

## Test attention

```bash
pixi run -e gpu python -m pytest tests -q
```

The existing attention tests run with BF16 CPU tensors. Flash tests run on CUDA
and skip when CUDA is unavailable. They exercise eager and compiled Flash,
compare against a float32 reference with BF16 tolerances, and check causality.
The float32 reference is only for testing, not part of the BF16 benchmarks.

## Pre-commit hooks

Activate the hooks once per checkout:

```bash
pixi run -e gpu pre-commit install
```

Before each commit, Ruff formats staged Python files and pytest runs the full
`tests/` suite in the GPU environment. CUDA tests skip if CUDA is unavailable.
If Ruff changes files, review and stage those changes, then retry the commit.
Test failures block the commit as well. No separate hook environments are used;
the hooks use the tools installed by Pixi.

To run both hooks manually against all tracked files:

```bash
pixi run -e gpu pre-commit run --all-files
```

## GitHub CI

`.github/workflows/ci.yaml` runs on every push to `main` and every pull request
targeting `main`, regardless of author. It installs only the locked Pixi `cpu`
environment and runs:

```bash
pixi run --locked -e cpu ruff format --check .
pixi run --locked -e cpu python -m pytest tests/test_model_cpu.py -q
```

The explicitly selected CPU test file covers configuration validation, attention
against an independent reference, built-in attention, and CPU compilation.
CUDA/Flash tests are not collected by CI. Local pre-commit still runs all tests.
CI checks formatting without modifying files; fix failures locally with Ruff.
Commit `pixi.lock` along with the workflow for reproducible installs.

To block merges when checks fail, configure a GitHub branch rule/ruleset for
`main` requiring the **Ruff format and CPU pytest** check. Running CI alone does
not enforce this merge restriction or prevent direct pushes.
