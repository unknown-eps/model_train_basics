# AGENTS.md

## Project goal

This is a **learning project**, not a production codebase. The user is building a small
Llama-like model from first principles to understand how it works. See `README.md` for
the objectives and step plan.

## How to work here

- Work **part by part**: complete one step, explain it, let the user absorb it before
  moving on. Do **not** one-shot the whole plan or dump a complete implementation.
- When you write code, explain *why* it works this way and how the pieces fit together.
- Follow the step order in `README.md` (env setup -> main code -> tests -> lint/pre-commit
  -> GitHub CI). Only move ahead when the current step is understood.

## Planned toolchain (from README.md — not yet set up)

The repo currently contains only `README.md`. The following is the agreed plan, not
verified config. As each piece lands, update this section with the real commands.

- **Pixi** for environment setup, with two groups: `gpu` (torch, transformers, ipykernel
  for VS Code notebook testing) and `cpu`-only for GitHub CI.
- **pytest** for tests, **ruff** for linting and formatting, enforced via pre-commit.
- Core tests must run on **CPU** so they work in GitHub CI. Flash Attention tests
  require CUDA and skip on CPU-only systems.
- End goal: integrate the model with HF `transformers` and enable/compare Flash
  Attention 2 against standard attention.

## Commands

- Run tests: `pixi run -e gpu python -m pytest tests -q` (core tests use BF16 CPU
  tensors; Flash tests use BF16 CUDA tensors and skip without CUDA).
- Run a single test: `pixi run -e gpu python -m pytest tests/test_model.py::test_attention_matches_loop_reference -q`.
- The same test commands can use `-e cpu` once that environment is installed.
- Format Python: `pixi run -e gpu ruff format .` (lint checking is not configured).
- Activate Git hooks: `pixi run -e gpu pre-commit install` (once per checkout).
- Run hooks: `pixi run -e gpu pre-commit run --all-files`. Ruff formats staged
  Python files on commit; pytest runs the full `tests/` suite on every commit.
