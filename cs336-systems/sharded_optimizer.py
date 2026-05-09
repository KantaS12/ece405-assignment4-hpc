"""ZeRO-1-style optimizer state sharding.

Each parameter is assigned to one rank in round-robin order. That owning rank
holds the optimizer state (Adam m, v, master weight copies, ...) for the
parameter; the other ranks hold no state for it. After each step, the owning
rank broadcasts the freshly-updated parameter back to every other rank so all
ranks finish the step with identical .data.

This is exactly the ZeRO-DP P_os scheme from Rajbhandari et al. (2020).
"""
from __future__ import annotations

from typing import Any, Iterable, Type

import torch
import torch.distributed as dist
from torch.optim import Optimizer


class ShardedOptimizer(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict],
        optimizer_cls: Type[Optimizer],
        **kwargs: Any,
    ) -> None:
        if dist.is_available() and dist.is_initialized():
            self._world_size = dist.get_world_size()
            self._rank = dist.get_rank()
        else:
            self._world_size = 1
            self._rank = 0

        # Materialise params (could be a generator or a list of {"params": [...]})
        param_list: list[torch.nn.Parameter] = []
        for entry in params:
            if isinstance(entry, dict):
                param_list.extend(entry["params"])
            else:
                param_list.append(entry)

        self._all_params: list[torch.nn.Parameter] = param_list
        # Round-robin assignment: parameter index i is owned by rank (i % world_size).
        self._owner_of: list[int] = [i % self._world_size for i in range(len(param_list))]
        self._owned_params: list[torch.nn.Parameter] = [
            p for p, owner in zip(param_list, self._owner_of) if owner == self._rank
        ]

        # Build the underlying optimizer over only the params this rank owns.
        if self._owned_params:
            self._inner = optimizer_cls(self._owned_params, **kwargs)
        else:
            self._inner = None
        self._kwargs = kwargs

        # torch.optim.Optimizer expects param_groups with all params — we expose the full
        # set so that callers (e.g. learning-rate schedulers) see every parameter.
        defaults = dict(kwargs)
        super().__init__(param_list, defaults)

    def add_param_group(self, param_group: dict) -> None:  # pragma: no cover - unused
        super().add_param_group(param_group)

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self._all_params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.detach_()
                    p.grad.zero_()

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 1. Each rank updates only its owned shard.
        if self._inner is not None:
            self._inner.step()

        # 2. Broadcast each parameter from its owner so every rank converges.
        if self._world_size > 1 and dist.is_initialized():
            for i, p in enumerate(self._all_params):
                dist.broadcast(p.data, src=self._owner_of[i])

        return loss

    # Convenience for benchmarking: how many bytes of optimizer state live on this rank?
    def owned_state_bytes(self) -> int:
        if self._inner is None:
            return 0
        total = 0
        for state in self._inner.state.values():
            for v in state.values():
                if torch.is_tensor(v):
                    total += v.element_size() * v.numel()
        return total
