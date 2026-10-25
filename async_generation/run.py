"""Run bounded HTTP generation against the mock server."""

import argparse
import asyncio
import time
from pathlib import Path

import httpx

from async_generation.random_prompt import generate_random_prompts
from async_generation.throughput import ThroughputMeter
from async_generation.writer import ShardWriter


async def request_worker(pending, client, results_queue):
    # Advancing this shared iterator has no await, so no other Task interrupts it.
    for prompt_id, prompt in pending:
        response = await client.post(
            "/generate", json={"id": prompt_id, "prompt": prompt}
        )
        response.raise_for_status()
        record = response.json()
        if record["id"] != prompt_id:
            raise ValueError("Response ID does not match prompt ID")
        await results_queue.put(record)


async def collect_results(results_queue, meter, writer):
    while True:
        record = await results_queue.get()
        if record is None:
            break
        meter.record(num_tokens=len(record["tokens"]))
        # Both Arrow conversion and the blocking write stay off the event loop.
        await asyncio.to_thread(writer.add, record)

    await asyncio.to_thread(writer.finish)
    meter.finish()


async def run(args):
    prompts = generate_random_prompts(seed=args.seed, num_prompts=args.num_prompts)
    pending = iter(prompts)
    results_queue = asyncio.Queue(maxsize=args.concurrency)
    writer = ShardWriter(args.output_dir, args.shard_size)

    async with httpx.AsyncClient(
        base_url=args.url,
        timeout=120.0,
        limits=httpx.Limits(max_connections=args.concurrency),
    ) as client:
        meter = ThroughputMeter(report_every=args.report_every)
        # A failure in either side cancels the other side instead of leaving
        # the collector waiting forever for a missing completion/end signal.
        async with asyncio.TaskGroup() as pipeline:
            pipeline.create_task(collect_results(results_queue, meter, writer))
            async with asyncio.TaskGroup() as workers:
                for _ in range(args.concurrency):
                    workers.create_task(request_worker(pending, client, results_queue))
            await results_queue.put(None)

    elapsed = time.perf_counter() - meter.start_time
    print(
        f"Finished: {meter.total_requests} requests, {meter.total_tokens} tokens, "
        f"{writer.shards_written} shards in {elapsed:.2f}s",
        flush=True,
    )


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--num-prompts", type=positive_int, default=4)
    parser.add_argument("--concurrency", type=positive_int, default=4)
    parser.add_argument("--shard-size", type=positive_int, default=2)
    parser.add_argument("--report-every", type=positive_int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    asyncio.run(run(parser.parse_args()))
