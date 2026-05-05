from __future__ import annotations

import os
from typing import Dict, Optional

import torch
import torch.distributed as dist


SIZE_PRESETS: Dict[str, dict] = {
    "small":  dict(d_model=768,  d_ff=3072,  num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096,  num_layers=24, num_heads=16),
    "large":  dict(d_model=1280, d_ff=5120,  num_layers=36, num_heads=20),
    "xl":     dict(d_model=1600, d_ff=6400,  num_layers=48, num_heads=25),
    "2.7B":   dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}


def setup_process_group(
    rank: int,
    world_size: int,
    backend: str = "gloo",
    master_addr: str = "127.0.0.1",
    master_port: str = "29501",
) -> torch.device:
    """Initialize the process group and return the device this rank should use."""
    # Always overwrite so re-using the same env across loop iterations works.
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    if backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL backend requires CUDA, but no GPU was found.")
        if world_size > torch.cuda.device_count():
            raise RuntimeError(
                f"NCCL refuses {world_size} ranks on {torch.cuda.device_count()} visible GPU(s). "
                "Use --backend gloo for single-GPU benchmarks."
            )
        local_rank = rank % torch.cuda.device_count()
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return device


def cleanup_process_group() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_master(rank: Optional[int] = None) -> bool:
    if rank is None:
        rank = dist.get_rank() if dist.is_initialized() else 0
    return rank == 0


def shard_batch(x: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    """Slice the leading (batch) dim across ranks. Assumes batch is divisible."""
    assert x.size(0) % world_size == 0, (
        f"Batch dim {x.size(0)} must be divisible by world_size {world_size}"
    )
    per_rank = x.size(0) // world_size
    start = rank * per_rank
    return x[start : start + per_rank]


def build_lm(size: str, vocab_size: int, context_length: int, rope_theta: float = 10000.0):
    """Construct a ``BasicsTransformerLM`` from the §1.1.2 preset table."""
    from cs336_basics.model import BasicsTransformerLM
    cfg = SIZE_PRESETS[size]
    return BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        rope_theta=rope_theta,
    )
