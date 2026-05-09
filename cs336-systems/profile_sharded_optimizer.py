"""Optimizer state sharding accounting — measures process RSS per rank
at three checkpoints (after model init, before optimizer.step, after optimizer.step)
for both vanilla AdamW and ShardedOptimizer(AdamW). Also prints the per-rank
bytes held in optimizer state.

We use process RSS (not torch.cuda.max_memory_allocated) because our ranks live on
CPU (Gloo). Single-process runs report only rank-0's RSS.
"""
from __future__ import annotations

import argparse
import os
import resource
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from distributed_utils import (
    SIZE_PRESETS,
    build_lm,
    cleanup_process_group,
    setup_process_group,
)
from sharded_optimizer import ShardedOptimizer


def _rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # ru_maxrss is KiB on Linux -> MiB


def _state_bytes(opt: torch.optim.Optimizer) -> int:
    total = 0
    for state in opt.state.values():
        for v in state.values():
            if torch.is_tensor(v):
                total += v.element_size() * v.numel()
    return total


def _worker(rank: int, world_size: int, args, out_path: str) -> None:
    setup_process_group(rank=rank, world_size=world_size, backend="gloo", master_port=args.master_port)
    torch.manual_seed(args.seed)

    raw_model = build_lm(args.size, args.vocab_size, args.context_length, args.rope_theta).to("cpu")
    rss_after_init = _rss_mib()

    if args.variant == "sharded":
        optimizer = ShardedOptimizer(raw_model.parameters(), torch.optim.AdamW, lr=1e-4)
    else:
        optimizer = torch.optim.AdamW(raw_model.parameters(), lr=1e-4)

    bs = args.batch_size
    x = torch.randint(0, args.vocab_size, (bs, args.context_length))
    y = torch.randint(0, args.vocab_size, (bs, args.context_length))
    loss_fn = nn.CrossEntropyLoss()

    # one full step
    optimizer.zero_grad(set_to_none=True)
    logits = raw_model(x)
    loss = loss_fn(logits.reshape(-1, args.vocab_size), y.reshape(-1))
    loss.backward()
    rss_before_step = _rss_mib()

    optimizer.step()
    rss_after_step = _rss_mib()
    state_bytes = _state_bytes(optimizer if args.variant != "sharded" else optimizer._inner) if not args.variant == "sharded" else (
        optimizer.owned_state_bytes()
    )

    line = (
        f"variant={args.variant} size={args.size} ws={world_size} rank={rank} "
        f"rss_after_init_MiB={rss_after_init:.1f} "
        f"rss_before_step_MiB={rss_before_step:.1f} "
        f"rss_after_step_MiB={rss_after_step:.1f} "
        f"state_bytes={state_bytes}"
    )
    print(line)
    if rank == 0 and out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "a") as f:
            f.write(line + "\n")
    # Gather all-rank lines through rank 0 too:
    gathered = [None] * world_size
    dist.all_gather_object(gathered, line)
    if rank == 0 and out_path:
        with open(out_path, "a") as f:
            for g in gathered:
                if g != line:
                    f.write(g + "\n")
    cleanup_process_group()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["dense", "sharded"], required=True)
    p.add_argument("--world-size", type=int, default=2)
    p.add_argument("--size", choices=list(SIZE_PRESETS), default="small")
    p.add_argument("--vocab-size", type=int, default=10000)
    p.add_argument("--context-length", type=int, default=128)
    p.add_argument("--rope-theta", type=float, default=10000.0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--master-port", default="29515")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    mp.spawn(_worker, args=(args.world_size, args, args.out), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
