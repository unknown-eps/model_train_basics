"""Compare full causal-attention forward passes on the local CUDA GPU."""

import torch

from model import attention_module, builtin_attention_module

# B, S, D, d_head; keep the explicit [B, H, S, S] scores modest for an 8 GB GPU.
SIZES = (
    (4, 128, 2048, 256),
    (4, 256, 2048, 256),
    (4, 512, 2048, 256),
    (4, 1024, 2048, 256),
    (4, 2048, 2048, 256),
)
WARMUP_STEPS = 25
REPETITIONS = 100


def peak_forward_memory(module, x):
    """Return peak extra allocated MiB, including output but excluding weights/input."""
    device = x.device
    torch.cuda.synchronize(device)
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    output = module(x)
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device) - baseline
    del output
    return peak / 1024**2


def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This benchmark requires a CUDA GPU; use the Pixi gpu environment."
        )

    torch.manual_seed(42)
    device = torch.device("cuda:0")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}")
    print("Mode: eval + inference_mode; dtype: float32; forward only; no torch.compile")
    print("Full modules timed, including projections and mask creation.")
    print("Both modules include matching Q/K/V and output projections.")
    print(
        "Memory: peak extra allocated MiB per forward (including output); excludes "
        "resident weights/inputs, allocator cache, and non-PyTorch CUDA allocations."
    )
    print(
        f"Timing: CUDA events; {WARMUP_STEPS} warmups then {REPETITIONS} measured "
        "forwards per module and size; alternating measurement order."
    )
    print("Speedup = custom mean / built-in mean (>1 means built-in is faster).\n")
    print(
        f"{'B/S/D/head':>20} {'Custom avg ms':>14} {'Built-in avg ms':>16} "
        f"{'Speedup':>10} {'Custom peak MiB':>17} {'Built-in peak MiB':>19}"
    )

    with torch.inference_mode():
        for B, S, D, d_head in SIZES:
            custom = (
                attention_module(D, d_head)
                .to(device=device, dtype=torch.float32)
                .eval()
            )
            builtin = builtin_attention_module.from_attention_module(custom).eval()
            x = torch.randn(B, S, D, device=device, dtype=torch.float32)

            modules = (custom, builtin)
            events = [
                [
                    (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    for _ in range(REPETITIONS)
                ]
                for _ in modules
            ]
            # Initialize events outside measurement: CUDA creates them lazily.
            for pairs in events:
                for start, end in pairs:
                    start.record()
                    end.record()
            torch.cuda.synchronize(device)

            # Warm both modules before timing either one.
            for _ in range(WARMUP_STEPS):
                for index in range(len(modules)):
                    modules[index](x)
            torch.cuda.synchronize(device)

            for repetition in range(REPETITIONS):
                order = (0, 1) if repetition % 2 == 0 else (1, 0)
                for index in order:
                    start, end = events[index][repetition]
                    start.record()
                    modules[index](x)
                    end.record()
            # Read elapsed GPU milliseconds only after all timed work completes.
            torch.cuda.synchronize(device)
            measurements = [
                sum(start.elapsed_time(end) for start, end in pairs) / REPETITIONS
                for pairs in events
            ]

            # Keep correctness checks outside both warmup and measured forwards.
            torch.testing.assert_close(builtin(x), custom(x), rtol=1e-4, atol=1e-5)

            # Profile each module separately, without disturbing the latency samples.
            custom_memory = peak_forward_memory(custom, x)
            builtin_memory = peak_forward_memory(builtin, x)

            custom_time, builtin_time = measurements
            shape = f"{B}/{S}/{D}/{d_head}"
            print(
                f"{shape:>20} {custom_time:14.3f} "
                f"{builtin_time:16.3f} "
                f"{custom_time / builtin_time:9.2f}x "
                f"{custom_memory:17.2f} {builtin_memory:19.2f}",
                flush=True,
            )
            del custom, builtin, modules, x


if __name__ == "__main__":
    main()
