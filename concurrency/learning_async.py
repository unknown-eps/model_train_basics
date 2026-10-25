import asyncio
import resource
import time


async def task(n):
    start = time.perf_counter()
    print(f"Task {n}: starting")
    # Pause this coroutine, not the event-loop thread.
    await asyncio.sleep(1)
    elapsed = time.perf_counter() - start
    print(f"Task {n}: finished after {elapsed:.3f} seconds (requested sleep: 1 second)")
    return n * n


async def compute_task():
    print("Compute: starting 2 seconds of CPU work; event loop is blocked")

    deadline = time.perf_counter() + 2
    count = 0
    # No await here: this coroutine keeps control until the loop finishes.
    while time.perf_counter() < deadline:
        count += 1

    print("Compute: finished; other coroutines can now resume")
    return count


async def main():
    # Linux/Unix process CPU counters; subtract to measure this run only.
    cpu_start = resource.getrusage(resource.RUSAGE_SELF)
    start = time.perf_counter()

    # Calling an async function creates a coroutine; it does not run it yet.
    coroutine_0 = task(0)
    coroutine_1 = task(1)
    coroutine_2 = task(2)
    coroutine_3 = task(3)
    coroutine_4 = task(4)
    compute_coroutine = compute_task()

    # Schedule compute first. Without an await in its body, it finishes before
    # any of the sleeping tasks can even start.
    results = await asyncio.gather(
        compute_coroutine,
        coroutine_0,
        coroutine_1,
        coroutine_2,
        coroutine_3,
        coroutine_4,
    )
    print(type(results))

    elapsed = time.perf_counter() - start
    cpu_end = resource.getrusage(resource.RUSAGE_SELF)
    user_cpu = cpu_end.ru_utime - cpu_start.ru_utime
    system_cpu = cpu_end.ru_stime - cpu_start.ru_stime
    print(f"Results: {results}")
    print(f"Total elapsed time: {elapsed:.3f} seconds")
    print(f"User CPU time: {user_cpu:.3f} seconds")
    print(f"System CPU time: {system_cpu:.3f} seconds")
    print(f"Total CPU time: {user_cpu + system_cpu:.3f} seconds")
    # 1.0 means one logical CPU fully occupied throughout the measured interval.
    cpu_fraction = (user_cpu + system_cpu) / elapsed
    print(f"CPU utilization fraction (one logical CPU = 1.0): {cpu_fraction:.3f}")
    print(f"CPU utilization: {cpu_fraction:.1%}")


if __name__ == "__main__":
    asyncio.run(main())
