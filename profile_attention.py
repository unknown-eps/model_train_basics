"""Inspect attention operators, CUDA kernels, and allocations separately from timing."""

import argparse
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from model import attention_module, builtin_attention_module


def profile_module(module, x, label: str, output_dir: Path):
    # Compile (if needed), then warm kernels and allocator outside the capture.
    for _ in range(25):
        module(x)
    torch.cuda.synchronize(x.device)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with record_function(label):
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
    args = parser.parse_args()
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
    print("Eager and torch.compile variants; backend=inductor, dynamic=False, mode=default.")
    print("Compilation occurs during warmup, outside profiler captures.")
    print("Profiling adds overhead; use benchmark.py for latency and peak memory.")
    print("Allocation tables show net operator allocations, not peak live memory.")
    print(f"Reports and Perfetto-compatible traces: {args.output_dir.resolve()}")

    with torch.inference_mode():
        for S in (1024, 2048):
            custom = attention_module(2048, 256).to(device=device, dtype=torch.float32).eval()
            builtin = builtin_attention_module.from_attention_module(custom).eval()
            x = torch.randn(4, S, 2048, device=device, dtype=torch.float32)
            variants = (
                ("custom", custom),
                ("builtin", builtin),
                ("custom_compiled", torch.compile(custom, backend="inductor", dynamic=False)),
                ("builtin_compiled", torch.compile(builtin, backend="inductor", dynamic=False)),
            )
            for name, module in variants:
                profile_module(module, x, f"{name}_s{S}", args.output_dir)
            del custom, builtin, variants, module, x


if __name__ == "__main__":
    main()
