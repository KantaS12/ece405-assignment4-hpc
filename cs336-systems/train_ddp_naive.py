from __future__ import annotations

import argparse
import os
import time
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from distributed_utils import (
    SIZE_PRESETS,
    build_lm,
    cleanup_process_group,
    is_master,
    setup_process_group,
    shard_batch,
)


class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def _all_reduce_grads(model: nn.Module, world_size: int) -> float:
    """All-reduce each parameter's grad and return seconds spent communicating."""
    if dist.get_backend() == "nccl":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for p in model.parameters():
        if p.grad is not None:
            p.grad.data.div_(world_size)
            dist.all_reduce(p.grad.data, async_op=False)
    if dist.get_backend() == "nccl":
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def _verify_worker(rank: int, world_size: int, backend: str, master_port: str):
    device = setup_process_group(rank, world_size, backend=backend, master_port=master_port)
    torch.manual_seed(0)
    # Single-process baseline
    baseline = _Toy().to(device)
    baseline_opt = torch.optim.SGD(baseline.parameters(), lr=0.1)

    # DDP copy
    ddp_model = deepcopy(baseline).to(device)
    # Broadcast just to mirror what real DDP does
    for p in ddp_model.parameters():
        dist.broadcast(p.data, src=0)
    ddp_opt = torch.optim.SGD(ddp_model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()

    torch.manual_seed(123)
    full_x = torch.randn(8, 8, device=device)
    full_y = torch.randn(8, 4, device=device)

    for step in range(5):
        # Single-process: forward+backward on the full batch.
        baseline_opt.zero_grad()
        bl_out = baseline(full_x)
        bl_loss = loss_fn(bl_out, full_y)
        bl_loss.backward()
        baseline_opt.step()

        # DDP: each rank gets its shard.
        ddp_opt.zero_grad()
        x_shard = shard_batch(full_x, rank, world_size)
        y_shard = shard_batch(full_y, rank, world_size)
        out = ddp_model(x_shard)
        loss = loss_fn(out, y_shard)
        loss.backward()
        _all_reduce_grads(ddp_model, world_size)
        ddp_opt.step()

    if rank == 0:
        for (n_b, p_b), (n_d, p_d) in zip(baseline.named_parameters(), ddp_model.named_parameters()):
            assert torch.allclose(p_b, p_d, atol=1e-5, rtol=1e-4), (
                f"naive_ddp mismatch on {n_b}: max |diff|={ (p_b - p_d).abs().max().item():.3e}"
            )
        print("naive_ddp verify: OK")
    cleanup_process_group()


def _bench_worker(rank, world_size, backend, master_port, args, out_path):
    device = setup_process_group(rank, world_size, backend=backend, master_port=master_port)
    torch.manual_seed(args.seed)

    model = build_lm(args.size, args.vocab_size, args.context_length, args.rope_theta).to(device)
    # Sync initial weights from rank 0.
    for p in model.parameters():
        dist.broadcast(p.data, src=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    # Synthetic batch 
    bs = args.batch_size
    assert bs % world_size == 0, "global batch must be divisible by world_size"
    full_x = torch.randint(0, args.vocab_size, (bs, args.context_length), device=device)
    full_y = torch.randint(0, args.vocab_size, (bs, args.context_length), device=device)

    iter_times = []
    comm_times = []
    for step in range(args.warmup + args.iters):
        optimizer.zero_grad(set_to_none=True)
        x_shard = shard_batch(full_x, rank, world_size)
        y_shard = shard_batch(full_y, rank, world_size)

        if device.type == "cuda":
            torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        logits = model(x_shard)
        loss = loss_fn(logits.reshape(-1, args.vocab_size), y_shard.reshape(-1))
        loss.backward()
        comm_s = _all_reduce_grads(model, world_size)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        iter_s = time.perf_counter() - t0

        if step >= args.warmup:
            iter_times.append(iter_s)
            comm_times.append(comm_s)

    # Aggregate across ranks.
    iter_mean_ms = (sum(iter_times) / len(iter_times)) * 1000.0
    comm_mean_ms = (sum(comm_times) / len(comm_times)) * 1000.0
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, (iter_mean_ms, comm_mean_ms))

    if rank == 0:
        iter_avg = sum(g[0] for g in gathered) / world_size
        comm_avg = sum(g[1] for g in gathered) / world_size
        msg = (
            f"naive_ddp_bench size={args.size} world_size={world_size} backend={backend} "
            f"bs={bs} ctx={args.context_length} iter_ms={iter_avg:.2f} comm_ms={comm_avg:.2f} "
            f"comm_frac={comm_avg/iter_avg:.3f}"
        )
        print(msg)
        if out_path:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            with open(out_path, "a") as f:
                f.write(msg + "\n")
    cleanup_process_group()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["verify", "bench"], default="bench")
    p.add_argument("--world-size", type=int, default=2)
    p.add_argument("--backend", choices=["gloo", "nccl"], default="gloo")
    p.add_argument("--master-port", default="29512")
    p.add_argument("--size", choices=list(SIZE_PRESETS), default="medium",
                   help="Model preset (default: medium; XL doesn't fit twice on 16GB).")
    p.add_argument("--vocab-size", type=int, default=10000)
    p.add_argument("--context-length", type=int, default=128)
    p.add_argument("--rope-theta", type=float, default=10000.0)
    p.add_argument("--batch-size", type=int, default=8, help="Global batch size.")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="Optional file to append results to.")
    args = p.parse_args()

    fn = _verify_worker if args.mode == "verify" else _bench_worker
    if args.mode == "verify":
        mp.spawn(fn, args=(args.world_size, args.backend, args.master_port), nprocs=args.world_size, join=True)
    else:
        mp.spawn(fn, args=(args.world_size, args.backend, args.master_port, args, args.out), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
