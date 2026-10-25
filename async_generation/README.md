# Async generation: mock server

Run from the repository root using the existing GPU environment:

```bash
pixi run -e gpu uvicorn async_generation.mock_vllm:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/docs to send requests interactively, or use:

```bash
curl http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"id": 0, "prompt": "Hello"}'
```

The response contains the input `id` and a uniformly sampled number of tokens
between 2048 and 4096, inclusive. Each token ID is in `[0, MAX_TOKEN_ID]`,
inclusive. This is a custom mock API, not an implementation
of the OpenAI/vLLM API.

The base random seed is `RANDOM_SEED = 42`. Each request uses its own generator
seeded with `RANDOM_SEED + id`, so the same ID and prompt reproduce the same
length and tokens regardless of concurrent request scheduling.

## Sequential generation

Each step waits 10 ms to simulate decoding latency, then computes:

```text
token = (previous_token + uniform_random_offset) % (MAX_TOKEN_ID + 1)
```

A request therefore takes roughly 20–41 seconds, plus scheduling overhead.

The next step uses this token as `previous_token`. Adding a fixed previous token
modulo the vocabulary size permutes the possible offsets, so each token is still
uniformly distributed even given the previous token. This creates a sequential
computational dependency, **not statistical correlation** between tokens.

The initial previous token is derived from the prompt's UTF-8 bytes. There is no
real model computation or GPU use. `asyncio.sleep` lets other requests progress
between decoding steps; a CPU-heavy Python loop would instead block the server's
event-loop thread.

Only the mock server is implemented at this stage; request workers and Parquet
sharding will come later.
