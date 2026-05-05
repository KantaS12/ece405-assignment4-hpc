from __future__ import annotations

import argparse
import csv
import os
import time
from typing import List

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from distributed_utils import setup_process_group, cleanup_process_group


SIZES_BYTES = {
    "1MB":   1 * 1024 * 1024,
    "10MB":  10 * 1024 * 1024,
    "100MB": 100 * 1024 * 1024,
    "1GB":   1 * 1024 * 1024 * 1024,
}


def _worker(rank: int, world_size: int, backend: str, sizes: List[str], iters: int, warmup: int, master_port: str, out_q):
    try:
        device = setup_process_group(rank, world_size, backend=backend, master_port=master_port)
    except Exception as e:
        if rank == 0:
            out_q.put({"error": f"setup failed: {e}"})
        return
    results = []
    for label in sizes:
        nbytes = SIZES_BYTES[label]
        n_floats = nbytes // 4
        x = torch.randn(n_floats, dtype=torch.float32, device=device)

        # Warmup
        for _ in range(warmup):
            dist.all_reduce(x, async_op=False)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dist.barrier()

        start = time.perf_counter()
        for _ in range(iters):
            dist.all_reduce(x, async_op=False)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dist.barrier()
        elapsed = time.perf_counter() - start
        per_call_ms = (elapsed / iters) * 1000.0

        # Aggregate timing across ranks (rank 0 reports the mean).
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, per_call_ms)
        if rank == 0:
            mean_ms = sum(gathered) / world_size
            results.append({
                "backend": backend,
                "device": device.type,
                "world_size": world_size,
                "size": label,
                "size_bytes": nbytes,
                "iters": iters,
                "per_call_ms": mean_ms,
            })

        # Drop the buffer before the next size so we don't OOM on 1GB x 6.
        del x
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if rank == 0:
        out_q.put(results)
    cleanup_process_group()


def run_one(backend: str, world_size: int, sizes: List[str], iters: int, warmup: int, master_port: str, timeout_s: int = 300):
    ctx = mp.get_context("spawn")
    out_q = ctx.Queue()
    procs = []
    for r in range(world_size):
        p = ctx.Process(target=_worker, args=(r, world_size, backend, sizes, iters, warmup, master_port, out_q))
        p.start()
        procs.append(p)
    try:
        results = out_q.get(timeout=timeout_s)
    except Exception as e:
        results = {"error": f"queue timeout / {e}"}
    for p in procs:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()
            p.join(timeout=5)
    if isinstance(results, dict) and "error" in results:
        print(f"  [skip] {backend} ws={world_size}: {results['error']}")
        return []
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backends", nargs="+", default=["gloo", "nccl"])
    p.add_argument("--world-sizes", type=int, nargs="+", default=[2, 4, 6])
    p.add_argument("--sizes", nargs="+", default=["1MB", "10MB", "100MB", "1GB"])
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--master-port", default="29510")
    p.add_argument("--csv-out", default=None)
    args = p.parse_args()

    all_rows = []
    base_port = int(args.master_port)
    port_offset = 0
    for backend in args.backends:
        if backend == "nccl" and not torch.cuda.is_available():
            print(f"[skip] backend=nccl requested but no GPU is visible.")
            continue
        for ws in args.world_sizes:
            if backend == "nccl" and torch.cuda.is_available() and ws > torch.cuda.device_count():
                print(f"[skip] nccl ws={ws} but only {torch.cuda.device_count()} GPU(s) visible "
                      f"(NCCL 2.21+ rejects duplicate GPUs on the same device).")
                continue
            port = str(base_port + port_offset)
            port_offset += 1
            print(f"[run] backend={backend} world_size={ws} port={port} sizes={args.sizes}")
            rows = run_one(backend, ws, args.sizes, args.iters, args.warmup, port)
            all_rows.extend(rows)
            for r in rows:
                print(f"  {r['backend']:5s} ws={r['world_size']} {r['size']:>5s}  {r['per_call_ms']:.3f} ms/call")

    if args.csv_out:
        os.makedirs(os.path.dirname(args.csv_out) or ".", exist_ok=True)
        with open(args.csv_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        print(f"wrote {args.csv_out}")


if __name__ == "__main__":
    main()
