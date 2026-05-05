from __future__ import annotations

from typing import Iterable, List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn


def _broadcast_module_state(module: nn.Module, src: int = 0) -> None:
    """Sync parameters and buffers from ``src`` to every other rank in-place."""
    for tensor in list(module.parameters()) + list(module.buffers()):
        dist.broadcast(tensor.data, src=src)


class DDPIndividualParameters(nn.Module):
    """All-reduce each parameter's gradient asynchronously as it becomes ready."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self._world_size = dist.get_world_size() if dist.is_initialized() else 1
        self._handles: List[dist.Work] = []

        # Make every rank start from rank-0 weights/buffers.
        if dist.is_initialized() and self._world_size > 1:
            _broadcast_module_state(self.module, src=0)

        self._hook_handles = []
        for p in self.module.parameters():
            if p.requires_grad:
                hook = p.register_post_accumulate_grad_hook(self._make_grad_hook())
                self._hook_handles.append(hook)

    def _make_grad_hook(self):
        def _hook(param: nn.Parameter):
            if self._world_size <= 1 or param.grad is None:
                return
            # Average gradients (sum / world_size) by scaling first so the
            # all-reduce just sums the already-scaled tensors.
            param.grad.data.div_(self._world_size)
            handle = dist.all_reduce(param.grad.data, async_op=True)
            self._handles.append(handle)
        return _hook

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        """Wait for every outstanding all-reduce to finish."""
        for h in self._handles:
            h.wait()
        self._handles.clear()


class _Bucket:
    """A group of parameters that get all-reduced together as one flat tensor."""

    __slots__ = ("params", "size_bytes", "remaining", "handle", "flat", "originals")

    def __init__(self) -> None:
        self.params: List[nn.Parameter] = []
        self.size_bytes: int = 0
        self.remaining: int = 0
        self.handle: Optional[dist.Work] = None
        self.flat: Optional[torch.Tensor] = None
        self.originals: Optional[List[torch.Tensor]] = None


class DDPBucketed(nn.Module):
    """All-reduce gradients in byte-budgeted buckets."""

    def __init__(self, module: nn.Module, bucket_size_mb: float):
        super().__init__()
        self.module = module
        self.bucket_size_bytes = int(bucket_size_mb * 1024 * 1024)
        self._world_size = dist.get_world_size() if dist.is_initialized() else 1

        if dist.is_initialized() and self._world_size > 1:
            _broadcast_module_state(self.module, src=0)

        self._buckets: List[_Bucket] = []
        self._param_to_bucket: dict = {}
        self._build_buckets()

        # Register Hooks
        self._hook_handles = []
        for p in self.module.parameters():
            if p.requires_grad:
                hook = p.register_post_accumulate_grad_hook(self._make_grad_hook())
                self._hook_handles.append(hook)

    def _build_buckets(self) -> None:
        # Reverse Parameter Order
        params_reversed = list(reversed(list(self.module.parameters())))
        cur = _Bucket()
        for p in params_reversed:
            if not p.requires_grad:
                continue
            nbytes = p.numel() * p.element_size()
            if cur.params and cur.size_bytes + nbytes > self.bucket_size_bytes:
                self._buckets.append(cur)
                cur = _Bucket()
            cur.params.append(p)
            cur.size_bytes += nbytes
            self._param_to_bucket[id(p)] = cur
        if cur.params:
            self._buckets.append(cur)

        for b in self._buckets:
            b.remaining = len(b.params)

    def _make_grad_hook(self):
        def _hook(param: nn.Parameter):
            if self._world_size <= 1 or param.grad is None:
                return
            bucket = self._param_to_bucket.get(id(param))
            if bucket is None:
                return
            bucket.remaining -= 1
            if bucket.remaining == 0:
                self._launch_bucket(bucket)
        return _hook

    def _launch_bucket(self, bucket: _Bucket) -> None:
        # Flatten the bucket's gradients into one big tensor, scale by world size for averaging, and all-reduce.
        grads = [p.grad.data for p in bucket.params]
        flat = torch._utils._flatten_dense_tensors(grads)
        flat.div_(self._world_size)
        bucket.flat = flat
        bucket.originals = grads
        bucket.handle = dist.all_reduce(flat, async_op=True)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        """Wait for in-flight bucket all-reduces and copy results back."""
        for b in self._buckets:
            if b.handle is None:
                continue
            b.handle.wait()
            unflat = torch._utils._unflatten_dense_tensors(b.flat, b.originals)
            for orig, new in zip(b.originals, unflat):
                orig.copy_(new)
            b.handle = None
            b.flat = None
            b.originals = None
            b.remaining = len(b.params)
