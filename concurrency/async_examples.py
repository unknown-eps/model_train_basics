import asyncio
import time


async def count():
    print("One")
    await asyncio.sleep(1)
    print("Two")
    await asyncio.sleep(1)
    print("Finished")


async def main():
    await asyncio.gather(count(), count(), count())


if __name__ == "__main__":
    start = time.perf_counter()
    asyncio.run(main())
    end = time.perf_counter()
    print(f"Elapsed time = {end - start}")
