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
    print("Mode: eval + inference_mode; dtype: float32; forward only")
    print("Compiled variants: torch.compile(backend='inductor', dynamic=False, mode='default').")
    print("Compilation occurs during warmup and is excluded from measurements.")
    print("Full modules timed, including projections and mask creation.")
    print("Both modules include matching Q/K/V and output projections.")
    print(
        "Memory: peak extra allocated MiB per forward (including output); excludes "
        "resident weights/inputs, allocator cache, and non-PyTorch CUDA allocations."
    )
    print(
        f"Timing: CUDA events; {WARMUP_STEPS} warmups then {REPETITIONS} measured "
        "forwards per module and size; rotating measurement order."
    )
    print("Speedup = eager custom mean / variant mean (>1 means faster).\n")
    print(
        f"{'B/S/D/head':>20} {'Variant':>20} {'Avg ms':>12} "
        f"{'Speedup':>10} {'Peak extra MiB':>16}"
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

            variants = (
                ("custom", custom),
                ("builtin", builtin),
                ("custom_compiled", torch.compile(custom, backend="inductor", dynamic=False)),
                ("builtin_compiled", torch.compile(builtin, backend="inductor", dynamic=False)),
            )
            modules = tuple(module for _, module in variants)
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

            # Compile and warm all modules before timing any of them.
            for _ in range(WARMUP_STEPS):
                for index in range(len(modules)):
                    modules[index](x)
            torch.cuda.synchronize(device)

            for repetition in range(REPETITIONS):
                for offset in range(len(modules)):
                    index = (repetition + offset) % len(modules)
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

            # Profile each module separately, without disturbing the latency samples.
            shape = f"{B}/{S}/{D}/{d_head}"
            for index in range(len(modules)):
                memory = peak_forward_memory(modules[index], x)
                print(
                    f"{shape:>20} {variants[index][0]:>20} {measurements[index]:12.3f} "
                    f"{measurements[0] / measurements[index]:9.2f}x {memory:16.2f}",
                    flush=True,
                )
            del custom, builtin, variants, modules, x


if __name__ == "__main__":
    main()
