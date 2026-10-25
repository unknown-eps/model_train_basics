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

## Run the client

With the server running, use another terminal at the repository root:

```bash
pixi run -e gpu python -m async_generation.run --output-dir async_generation/output/run-001
```

The output directory must not already exist, to avoid overwriting previous runs.
The defaults are four prompts, four concurrent requests, two records per shard,
and a throughput report every 1000 completions (plus the final remainder).
Use `--num-prompts`, `--concurrency`, `--shard-size`, `--report-every`, and
`--seed` to change these settings.

One collector measures completion throughput, buffers records, and awaits a
background-thread writer. Files have an explicit `int32` / `list<int32>` schema
and Zstd compression. Temporary files are renamed only after a successful close;
this is not a guarantee of durability against power loss.

The initial failure policy is fail-fast: a failed request or writer cancels the
pipeline. Previously saved shards remain; queued/unbuffered work can be lost.
There are no retries or resume support yet. Cancelling an await on a writer
thread does not forcibly stop a write already running in that thread.
