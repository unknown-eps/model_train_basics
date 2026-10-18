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
