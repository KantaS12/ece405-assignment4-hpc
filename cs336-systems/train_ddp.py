from __future__ import annotations

import argparse
import os
import time

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
    setup_process_group,
    shard_batch,
)
from ddp import DDPBucketed, DDPIndividualParameters


def _flatten_grads(params):
    return torch._utils._flatten_dense_tensors([p.grad.data for p in params if p.grad is not None])


def _unflatten_into_grads(flat, params):
    grads = [p.grad.data for p in params if p.grad is not None]
    unflat = torch._utils._unflatten_dense_tensors(flat, grads)
    for g, u in zip(grads, unflat):
        g.copy_(u)


def _comm_individual(model, world_size):
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


def _comm_flat(model, world_size):
    params = [p for p in model.parameters() if p.grad is not None]
    if dist.get_backend() == "nccl":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    flat = _flatten_grads(params)
    flat.div_(world_size)
    dist.all_reduce(flat, async_op=False)
    _unflatten_into_grads(flat, params)
    if dist.get_backend() == "nccl":
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def _wait_overlap_or_bucketed(ddp_model):
    if dist.get_backend() == "nccl":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    ddp_model.finish_gradient_synchronization()
    if dist.get_backend() == "nccl":
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def _wrap_model(variant: str, model: nn.Module, bucket_size_mb: float):
    if variant == "overlap":
        return DDPIndividualParameters(model)
    if variant == "bucketed":
        return DDPBucketed(model, bucket_size_mb=bucket_size_mb)
    return model  # individual / flat use the bare model.


def _broadcast_initial_weights(model: nn.Module):
    for p in model.parameters():
        dist.broadcast(p.data, src=0)


def _worker(rank, world_size, args, out_path):
    device = setup_process_group(rank, world_size, backend=args.backend, master_port=args.master_port)
    torch.manual_seed(args.seed)

    raw_model = build_lm(args.size, args.vocab_size, args.context_length, args.rope_theta).to(device)
    if args.variant in ("individual", "flat"):
        _broadcast_initial_weights(raw_model)
    model = _wrap_model(args.variant, raw_model, args.bucket_size_mb)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    bs = args.batch_size
    assert bs % world_size == 0
    full_x = torch.randint(0, args.vocab_size, (bs, args.context_length), device=device)
    full_y = torch.randint(0, args.vocab_size, (bs, args.context_length), device=device)

    iter_times, comm_times = [], []
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

        if args.variant == "individual":
            comm_s = _comm_individual(raw_model, world_size)
        elif args.variant == "flat":
            comm_s = _comm_flat(raw_model, world_size)
        else:
            comm_s = _wait_overlap_or_bucketed(model)

        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        iter_s = time.perf_counter() - t0

        if step >= args.warmup:
            iter_times.append(iter_s)
            comm_times.append(comm_s)

    iter_mean_ms = sum(iter_times) / len(iter_times) * 1000.0
    comm_mean_ms = sum(comm_times) / len(comm_times) * 1000.0
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, (iter_mean_ms, comm_mean_ms))
    if rank == 0:
        iter_avg = sum(g[0] for g in gathered) / world_size
        comm_avg = sum(g[1] for g in gathered) / world_size
        bucket_str = f" bucket_mb={args.bucket_size_mb}" if args.variant == "bucketed" else ""
        msg = (
            f"variant={args.variant} size={args.size} ws={world_size} backend={args.backend} "
            f"bs={bs} ctx={args.context_length}{bucket_str} "
            f"iter_ms={iter_avg:.2f} comm_ms={comm_avg:.2f} comm_frac={comm_avg/iter_avg:.3f}"
        )
        print(msg)
        if out_path:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            with open(out_path, "a") as f:
                f.write(msg + "\n")
    cleanup_process_group()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=["individual", "flat", "overlap", "bucketed"], required=True)
    p.add_argument("--world-size", type=int, default=2)
    p.add_argument("--backend", choices=["gloo", "nccl"], default="gloo")
    p.add_argument("--master-port", default="29513")
    p.add_argument("--size", choices=list(SIZE_PRESETS), default="medium")
    p.add_argument("--vocab-size", type=int, default=10000)
    p.add_argument("--context-length", type=int, default=128)
    p.add_argument("--rope-theta", type=float, default=10000.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--bucket-size-mb", type=float, default=10.0)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    mp.spawn(_worker, args=(args.world_size, args, args.out), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
