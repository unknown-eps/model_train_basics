"""Inspect attention operators, CUDA kernels, and allocations separately from timing."""

import argparse
from pathlib import Path
import subprocess
import sys

import torch
from torch.profiler import ProfilerActivity, profile

from model import attention_module, builtin_attention_module

VARIANTS = ("custom", "builtin", "custom_compiled", "builtin_compiled")
SEQUENCE_LENGTHS = (1024, 2048)


def profile_module(module, x, label: str, output_dir: Path, with_stack: bool):
    # Compile (if needed), then warm kernels and allocator outside the capture.
    for _ in range(25):
        module(x)
    torch.cuda.synchronize(x.device)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=with_stack,
    ) as prof:
        # The filename identifies this capture. An extra record_function range
        # can partially overlap its Python entry frame in Perfetto.
        output = module(x)
        torch.cuda.synchronize(x.device)
        del output

    averages = prof.key_averages(group_by_input_shape=True)
    timing = averages.table(sort_by="self_device_time_total", row_limit=20)
    memory = averages.table(sort_by="self_device_memory_usage", row_limit=20)
    # Include projections, copies, and allocations, not only attention-named ops.
    operators = sorted(
        {
            event.key
            for event in averages
            if event.key.startswith("aten::")
        }
    )
    # Actual kernel names also expose generated Triton and library CUDA kernels.
    kernels = sorted(
        {kernel.name for event in prof.events() for kernel in (event.kernels or ())}
    )
    operator_list = "\n".join(f"  {name}" for name in operators) or "  None recorded"
    kernel_list = "\n".join(f"  {name}" for name in kernels) or "  None recorded"
    report = (
        f"{label}\nAll ATen operators:\n{operator_list}\n\n"
        f"CUDA kernels/device activities (unique names):\n{kernel_list}\n\n"
        f"Top device time:\n{timing}\n\nTop net device allocations:\n{memory}\n"
    )
    (output_dir / f"{label}.txt").write_text(report)
    prof.export_chrome_trace(str(output_dir / f"{label}.json"))
    print(report, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=Path("profiles"))
    parser.add_argument("--variant", choices=VARIANTS, help="Profile only this variant.")
    parser.add_argument("--seq_len", type=int, choices=SEQUENCE_LENGTHS)
    parser.add_argument(
        "--with_stack",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record Python call stacks (default: enabled); adds detail and overhead.",
    )
    args = parser.parse_args()

    # Each capture starts a fresh interpreter, with no workers from other variants.
    if args.variant is None or args.seq_len is None:
        variants = (args.variant,) if args.variant else VARIANTS
        lengths = (args.seq_len,) if args.seq_len else SEQUENCE_LENGTHS
        for S in lengths:
            for variant in variants:
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--variant", variant,
                    "--seq_len", str(S),
                    "--output_dir", str(args.output_dir.resolve()),
                ]
                command.append("--with_stack" if args.with_stack else "--no-with_stack")
                subprocess.run(command, check=True)
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; use the Pixi gpu environment.")
    if ProfilerActivity.CUDA not in torch.profiler.supported_activities():
        raise RuntimeError("CUDA profiling is unavailable; check PyTorch/CUPTI support.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    device = torch.device("cuda:0")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}")
    print("B=4, D=2048, heads=8; float32; eval + inference_mode")
    print(f"Isolated capture: {args.variant}, S={args.seq_len}; Python stacks={args.with_stack}")
    print("Compiled variants use backend=inductor, dynamic=False, mode=default.")
    print("Compilation occurs during warmup, outside profiler captures.")
    print("Profiling adds overhead; use benchmark.py for latency and peak memory.")
    print("Allocation tables show net operator allocations, not peak live memory.")
    print(f"Reports and Perfetto-compatible traces: {args.output_dir.resolve()}")

    with torch.inference_mode():
        custom = attention_module(2048, 256).to(device=device, dtype=torch.float32).eval()
        x = torch.randn(4, args.seq_len, 2048, device=device, dtype=torch.float32)
        if args.variant.startswith("builtin"):
            module = builtin_attention_module.from_attention_module(custom).eval()
            del custom
        else:
            module = custom
        if args.variant.endswith("_compiled"):
            module = torch.compile(module, backend="inductor", dynamic=False)
        profile_module(
            module, x, f"{args.variant}_s{args.seq_len}", args.output_dir, args.with_stack
        )


if __name__ == "__main__":
    main()
