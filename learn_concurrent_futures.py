import argparse
import time
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor


def task(n, base_time, use_diff=False):
    if use_diff:
        base_time = base_time + n
    # Simulate I/0
    time.sleep(base_time)
    return n * n


def compute_task(n):
    """Pure Python computation to compare thread and process pools."""
    total = 0
    for i in range(10_000_000):
        total += (i + n) % 97
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compute", action="store_true", help="Run CPU work instead of sleep")
    parser.add_argument("--process_pool",action="store_true",help="Use process pool executor")
    args = parser.parse_args()
    start = time.perf_counter()
    my_executor = ProcessPoolExecutor if args.process_pool else ThreadPoolExecutor
    with my_executor(max_workers=5) as executor:
        if args.compute:
            futures = [executor.submit(compute_task, n) for n in range(5)]
        else:
            futures = [executor.submit(task, n, base_time=1) for n in range(5)]
        results = [future.result() for future in futures]

    elapsed = time.perf_counter() - start
    print(f"Results: {results}")
    print(f"Total elapsed time: {elapsed:.3f} seconds")
