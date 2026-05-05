import argparse
import csv
import gc
import math
import time
from typing import List

import torch
from einops import einsum

from cs336_basics.nn_utils import softmax


DIMS_DEFAULT = [16, 32, 64, 128]
SEQS_DEFAULT = [256, 1024, 4096, 8192, 16384]


def attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    d_k = K.shape[-1]
    scores = einsum(Q, K, "... q d, ... k d -> ... q k") / math.sqrt(d_k)
    weights = softmax(scores, dim=-1)
    return einsum(weights, V, "... q k, ... k d -> ... q d")


def is_oom(err: BaseException) -> bool:
    if isinstance(err, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(err, RuntimeError) and "out of memory" in str(err).lower()


def bench_one(batch: int, seq: int, dim: int, n_iters: int, warmup: int, device: str) -> dict:
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    Q = torch.randn(batch, seq, dim, device=device, requires_grad=True)
    K = torch.randn(batch, seq, dim, device=device, requires_grad=True)
    V = torch.randn(batch, seq, dim, device=device, requires_grad=True)

    for _ in range(warmup):
        out = attention(Q, K, V)
        out.sum().backward()
    torch.cuda.synchronize()

    fwd_times: List[float] = []
    for _ in range(n_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = attention(Q, K, V)
        torch.cuda.synchronize()
        fwd_times.append((time.perf_counter() - t0) * 1000.0)
        del out

    out = attention(Q, K, V)
    loss = out.sum()
    torch.cuda.synchronize()
    mem_before_bwd = torch.cuda.memory_allocated()

    bwd_times: List[float] = []
    for _ in range(n_iters):
        out = attention(Q, K, V)
        loss = out.sum()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        bwd_times.append((time.perf_counter() - t0) * 1000.0)

    peak = torch.cuda.max_memory_allocated()
    return {
        "fwd_mean_ms": sum(fwd_times) / len(fwd_times),
        "fwd_min_ms": min(fwd_times),
        "bwd_mean_ms": sum(bwd_times) / len(bwd_times),
        "bwd_min_ms": min(bwd_times),
        "mem_before_bwd_MiB": mem_before_bwd / (1024 ** 2),
        "peak_MiB": peak / (1024 ** 2),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--dims", type=int, nargs="+", default=DIMS_DEFAULT)
    p.add_argument("--seqs", type=int, nargs="+", default=SEQS_DEFAULT)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--csv-out", default=None,
                   help="Optional path to write a CSV of (dim, seq, fwd_ms, bwd_ms, mem_MiB, status).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.device != "cuda":
        raise SystemExit("This benchmark is GPU-only; pass --device cuda on a CUDA node.")

    print(f"batch={args.batch_size} iters={args.iters} warmup={args.warmup} device={args.device}")
    print(f"{'dim':>4} {'seq':>6}  {'fwd ms':>9} {'bwd ms':>9} {'mem MiB':>10} {'peak MiB':>10}  status")

    rows: List[dict] = []
    for dim in args.dims:
        for seq in args.seqs:
            try:
                r = bench_one(args.batch_size, seq, dim, args.iters, args.warmup, args.device)
                print(f"{dim:>4} {seq:>6}  {r['fwd_mean_ms']:>9.3f} {r['bwd_mean_ms']:>9.3f} "
                      f"{r['mem_before_bwd_MiB']:>10.1f} {r['peak_MiB']:>10.1f}  ok")
                rows.append(dict(dim=dim, seq=seq, status="ok", **r))
            except BaseException as e:  # OOM or other
                if is_oom(e):
                    print(f"{dim:>4} {seq:>6}  {'-':>9} {'-':>9} {'-':>10} {'-':>10}  OOM")
                    rows.append(dict(dim=dim, seq=seq, status="OOM"))
                    torch.cuda.empty_cache()
                    gc.collect()
                else:
                    raise

    if args.csv_out:
        keys = ["dim", "seq", "status", "fwd_mean_ms", "fwd_min_ms",
                "bwd_mean_ms", "bwd_min_ms", "mem_before_bwd_MiB", "peak_MiB"]
        with open(args.csv_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in keys})
        print(f"\nwrote {args.csv_out}")


if __name__ == "__main__":
    main()
