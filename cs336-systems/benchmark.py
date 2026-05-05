import argparse
import time
from contextlib import nullcontext
from typing import List

import torch
import torch.cuda.nvtx as nvtx

import cs336_basics.model
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

from implementation import annotated_scaled_dot_product_attention

cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

"""
Benchmark utilities

def mean(x: List[float]) -> float:
    return sum(x) / len(x)

# Benchmark that runs the provided function
def benchmark(description: str, run: Callable, num_warmups: int, num_trials: int) -> None:
    for _ in range(num_warmups):
        run()
        if torch.cuda.is_available():
            torch.cuda.synchronize() # Barrier that waits for every thread to finish

    times: List[float] = []
    for _ in range(num_trials):
        start = timeit.default_timer()
        run()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((timeit.default_timer() - start) * 1000)

    sorted_times = [round(t, 1) for t in sorted(times)]
    print(f"{description}: {sorted_times} (mean {round(mean(times), 1)} ms, "
          f"min {round(min(times), 1)} ms, max {round(max(times), 1)} ms)")

"""

SIZE_PRESETS = {
    "small":  dict(d_model=768,  d_ff=3072,  num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096,  num_layers=24, num_heads=16),
    "large":  dict(d_model=1280, d_ff=5120,  num_layers=36, num_heads=20),
    "xl":     dict(d_model=1600, d_ff=6400,  num_layers=48, num_heads=25),
    "2.7B":   dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="End-to-end forward/backward benchmark for BasicsTransformerLM.")
    p.add_argument("--vocab-size", type=int, default=10000)
    p.add_argument("--context-length", type=int, default=128)
    p.add_argument("--d-model", type=int, default=768)
    p.add_argument("--num-layers", type=int, default=12)
    p.add_argument("--num-heads", type=int, default=12)
    p.add_argument("--d-ff", type=int, default=3072)
    p.add_argument("--rope-theta", type=float, default=10000.0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--mode", choices=["forward", "forward_backward"], default="forward_backward")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--size", choices=list(SIZE_PRESETS), default=None,
                   help="§1.1.2 preset; overrides --d-model/--d-ff/--num-layers/--num-heads")
    p.add_argument("--mixed-precision", action="store_true",
                   help="Wrap forward in torch.autocast(bfloat16). Backward inherits the autocasted graph.")
    p.add_argument("--memory-profile", action="store_true",
                   help="Record a CUDA memory snapshot around the timed steps and dump it to --memory-snapshot-out.")
    p.add_argument("--memory-snapshot-out", default="memory_snapshot.pickle",
                   help="Path to write the pickle consumed by https://pytorch.org/memory_viz")
    p.add_argument("--compile", action="store_true",
                   help="Wrap the model with torch.compile after construction.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.size is not None:
        for k, v in SIZE_PRESETS[args.size].items():
            setattr(args, k, v)

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(device)

    if args.compile:
        model = torch.compile(model)

    inputs = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)
    targets = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)

    optimizer = AdamW(model.parameters(), lr=1e-4)

    def amp_ctx():
        if args.mixed_precision and device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def run_forward() -> None:
        with torch.no_grad(), nvtx.range("forward"), amp_ctx():
            model(inputs)

    def run_forward_backward() -> None:
        with nvtx.range("zero_grad"):
            optimizer.zero_grad(set_to_none=True)
        with nvtx.range("forward"), amp_ctx():
            logits = model(inputs)
            loss = cross_entropy(logits, targets)
        with nvtx.range("backward"):
            loss.backward()
        with nvtx.range("optimizer"):
            optimizer.step()

    if args.mode == "forward":
        model.eval()
        run = run_forward
    else:
        model.train()
        run = run_forward_backward

    description = (
        f"{args.mode} | size={args.size or 'custom'} layers={args.num_layers} d_model={args.d_model} "
        f"heads={args.num_heads} d_ff={args.d_ff} ctx={args.context_length} "
        f"batch={args.batch_size} amp={'bf16' if args.mixed_precision else 'fp32'} "
        f"compile={'on' if args.compile else 'off'} device={device.type}"
    )
    print(description)

    for _ in range(args.warmup):
        run()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    if args.memory_profile and device.type == "cuda":
        torch.cuda.memory._record_memory_history(max_entries=1000000)

    times: List[float] = []
    if device.type == "cuda":
        torch.cuda.cudart().cudaProfilerStart()
    try:
        for i in range(args.steps):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with nvtx.range(f"step {i}"):
                run()
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
    finally:
        if device.type == "cuda":
            torch.cuda.cudart().cudaProfilerStop()
        if args.memory_profile and device.type == "cuda":
            torch.cuda.memory._dump_snapshot(args.memory_snapshot_out)
            torch.cuda.memory._record_memory_history(enabled=None)
            print(f"memory snapshot written to {args.memory_snapshot_out}")

    times_sorted = sorted(times)
    mean = sum(times) / len(times)
    print(f"step times (ms): {[round(t, 2) for t in times_sorted]}")
    print(f"  mean {mean:.2f}  min {min(times):.2f}  max {max(times):.2f}  n={len(times)}")
    if device.type == "cuda":
        peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
        peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
        print(f"  peak_allocated {peak_alloc:.1f} MiB  peak_reserved {peak_reserved:.1f} MiB")


if __name__ == "__main__":
    main()


# Sweep (run on a GPU node):
#
#   for SZ in small medium large xl 2.7B; do
#     for AMP in "" "--mixed-precision"; do
#       uv run python cs336-systems/benchmark.py \
#           --size $SZ --mode forward_backward --warmup 2 --steps 10 $AMP
#     done
#   done
#
# First Benchmark Results

# uv run python cs336-systems/benchmark.py --warmup 5 --steps 10
# forward_backward | layers=12 d_model=768 heads=12 d_ff=3072 ctx=128 batch=4 device=cpu: [19294.3, 19482.4, 19561.2, 19566.8, 19839.3, 22408.9, 23571.8, 23825.0, 25232.5, 25660.0] (mean 21844.2 ms, min 19294.3 ms, max 25660.0 ms)

# uv run python cs336-systems/benchmark.py --warmup 0 --steps 10
# forward_backward | layers=12 d_model=768 heads=12 d_ff=3072 ctx=128 batch=4 device=cpu: [18937.3, 19059.1, 19105.4, 19124.4, 19132.3, 19143.6, 19237.2, 19344.5, 19441.8, 21418.3] (mean 19394.4 ms, min 18937.3 ms, max 21418.3 ms)

# uv run python cs336-systems/benchmark.py --warmup 1 --steps 10
# forward_backward | layers=12 d_model=768 heads=12 d_ff=3072 ctx=128 batch=4 device=cpu: [18983.2, 19009.0, 19037.5, 19069.9, 19112.6, 19169.5, 19173.6, 19228.1, 19269.8, 19272.0] (mean 19132.5 ms, min 18983.2 ms, max 19272.0 ms)

# uv run python cs336-systems/benchmark.py --warmup 2 --steps 10
# forward_backward | layers=12 d_model=768 heads=12 d_ff=3072 ctx=128 batch=4 device=cpu: [18863.0, 18880.7, 18912.3, 18913.1, 19060.4, 19064.0, 19104.5, 19155.3, 19212.4, 19456.7] (mean 19062.2 ms, min 18863.0 ms, max 19456.7 ms)


# nsys_profile

# module load system/CUDA/12.9.1
