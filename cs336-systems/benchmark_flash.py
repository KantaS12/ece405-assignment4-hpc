from __future__ import annotations

import argparse
import csv
import gc
import math
import pathlib
import sys
from typing import Callable, List, Tuple

import torch
from einops import einsum

import triton.testing as triton_testing


SEQS_DEFAULT = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
DIMS_DEFAULT = [16, 32, 64, 128]
DTYPES_DEFAULT = ["bf16", "fp32"]
DTYPE_MAP = {"bf16": torch.bfloat16, "fp32": torch.float32}


def attention_regular(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool) -> torch.Tensor:
    d_k = K.shape[-1]
    scale = 1.0 / math.sqrt(d_k)
    S = einsum(Q, K, "... q d, ... k d -> ... q k") * scale
    if is_causal:
        Nq, Nk = Q.shape[-2], K.shape[-2]
        q_idx = torch.arange(Nq, device=Q.device)[None, :, None]
        k_idx = torch.arange(Nk, device=Q.device)[None, None, :]
        S = torch.where(q_idx >= k_idx, S, S.new_full((), -1e6))
    P = torch.softmax(S, dim=-1)
    return einsum(P, V, "... q k, ... k d -> ... q d")


def discover_impls() -> List[Tuple[str, Callable]]:
    impls: List[Tuple[str, Callable]] = [("torch", attention_regular)]

    here = pathlib.Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))

    try:
        from flash_attention import FlashAttentionPytorch
        impls.append(("flash-pytorch", FlashAttentionPytorch.apply))
    except Exception as e:  # noqa: BLE001
        print(f"[skip] FlashAttentionPytorch: {e}", file=sys.stderr)

    try:
        from flash_attention_triton import FlashAttentionTriton  # type: ignore
        impls.append(("flash-triton", FlashAttentionTriton.apply))
    except Exception as e:  # noqa: BLE001
        print(f"[skip] FlashAttentionTriton: {e}", file=sys.stderr)

    return impls


def is_oom(err: BaseException) -> bool:
    if isinstance(err, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(err, RuntimeError) and "out of memory" in str(err).lower()


def bench_one(impl_fn: Callable, batch: int, seq: int, d: int, dtype: torch.dtype,
              is_causal: bool, device: str, warmup_ms: float, rep_ms: float) -> dict:
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    Q = torch.randn(batch, seq, d, device=device, dtype=dtype, requires_grad=True)
    K = torch.randn(batch, seq, d, device=device, dtype=dtype, requires_grad=True)
    V = torch.randn(batch, seq, d, device=device, dtype=dtype, requires_grad=True)

    def fwd_no_grad():
        with torch.no_grad():
            impl_fn(Q, K, V, is_causal)

    def fwd_bwd():
        out = impl_fn(Q, K, V, is_causal)
        out.sum().backward()

    fwd_ms = triton_testing.do_bench(fwd_no_grad, warmup=warmup_ms, rep=rep_ms)
    e2e_ms = triton_testing.do_bench(fwd_bwd, warmup=warmup_ms, rep=rep_ms,
                                     grad_to_none=[Q, K, V])
    bwd_ms = max(e2e_ms - fwd_ms, 0.0)

    return {
        "fwd_ms": fwd_ms,
        "bwd_ms": bwd_ms,
        "e2e_ms": e2e_ms,
        "peak_MiB": torch.cuda.max_memory_allocated() / (1024 ** 2),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seqs", type=int, nargs="+", default=SEQS_DEFAULT)
    p.add_argument("--dims", type=int, nargs="+", default=DIMS_DEFAULT)
    p.add_argument("--dtypes", nargs="+", choices=list(DTYPE_MAP), default=DTYPES_DEFAULT)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--no-causal", action="store_true",
                   help="Disable causal masking (default is is_causal=True per assignment).")
    p.add_argument("--warmup-ms", type=float, default=25.0,
                   help="triton.testing.do_bench warmup window in ms.")
    p.add_argument("--rep-ms", type=float, default=100.0,
                   help="triton.testing.do_bench measurement window in ms.")
    p.add_argument("--only", nargs="+", default=None,
                   help="Restrict to these implementations (e.g. torch flash-pytorch).")
    p.add_argument("--skip-impl-on-oom", action="store_true",
                   help="Once an impl OOMs at a given (dtype, dim), skip larger seqs for it.")
    p.add_argument("--csv-out", default=None, help="Optional CSV path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required: triton.testing.do_bench targets GPU.")

    impls = discover_impls()
    if args.only:
        impls = [(n, fn) for (n, fn) in impls if n in args.only]
    if not impls:
        raise SystemExit("No attention implementations available to benchmark.")

    is_causal = not args.no_causal
    print(f"impls: {[n for n, _ in impls]}  causal={is_causal}  batch={args.batch_size}")
    header = f"{'dtype':>5} {'dim':>4} {'seq':>6}  {'impl':>14}  {'fwd ms':>9} {'bwd ms':>9} {'e2e ms':>9} {'peak MiB':>10}  status"
    print(header)

    rows: List[dict] = []
    skipped: set = set()  # (impl_name, dtype, dim)

    for dtype_name in args.dtypes:
        dtype = DTYPE_MAP[dtype_name]
        for d in args.dims:
            for seq in args.seqs:
                for name, fn in impls:
                    key = (name, dtype_name, d)
                    if args.skip_impl_on_oom and key in skipped:
                        print(f"{dtype_name:>5} {d:>4} {seq:>6}  {name:>14}  "
                              f"{'-':>9} {'-':>9} {'-':>9} {'-':>10}  skip(prevOOM)")
                        rows.append(dict(dtype=dtype_name, dim=d, seq=seq, impl=name, status="skip"))
                        continue
                    try:
                        r = bench_one(fn, args.batch_size, seq, d, dtype, is_causal,
                                      "cuda", args.warmup_ms, args.rep_ms)
                        print(f"{dtype_name:>5} {d:>4} {seq:>6}  {name:>14}  "
                              f"{r['fwd_ms']:>9.3f} {r['bwd_ms']:>9.3f} {r['e2e_ms']:>9.3f} "
                              f"{r['peak_MiB']:>10.1f}  ok")
                        rows.append(dict(dtype=dtype_name, dim=d, seq=seq, impl=name,
                                         status="ok", **r))
                    except BaseException as e:  # noqa: BLE001
                        if is_oom(e):
                            print(f"{dtype_name:>5} {d:>4} {seq:>6}  {name:>14}  "
                                  f"{'-':>9} {'-':>9} {'-':>9} {'-':>10}  OOM")
                            rows.append(dict(dtype=dtype_name, dim=d, seq=seq, impl=name, status="OOM"))
                            skipped.add(key)
                            torch.cuda.empty_cache()
                            gc.collect()
                        else:
                            raise

    if args.csv_out:
        keys = ["dtype", "dim", "seq", "impl", "status", "fwd_ms", "bwd_ms", "e2e_ms", "peak_MiB"]
        with open(args.csv_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in keys})
        print(f"\nwrote {args.csv_out}")


if __name__ == "__main__":
    main()
