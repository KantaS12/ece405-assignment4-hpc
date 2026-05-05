from __future__ import annotations

import math

import torch


class FlashAttentionPytorch(torch.autograd.Function):
    Q_TILE_SIZE: int = 16
    K_TILE_SIZE: int = 16

    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        Bq = FlashAttentionPytorch.Q_TILE_SIZE
        Bk = FlashAttentionPytorch.K_TILE_SIZE

        *batch_dims, Nq, d = Q.shape
        Nk = K.shape[-2]
        scale = 1.0 / math.sqrt(d)

        Qf = Q.reshape(-1, Nq, d)
        Kf = K.reshape(-1, Nk, d)
        Vf = V.reshape(-1, Nk, d)
        B = Qf.shape[0]

        out_dtype = Q.dtype
        compute_dtype = torch.float32

        O = torch.empty(B, Nq, d, device=Q.device, dtype=out_dtype)
        L = torch.empty(B, Nq, device=Q.device, dtype=compute_dtype)

        Tq = (Nq + Bq - 1) // Bq
        Tk = (Nk + Bk - 1) // Bk

        for i in range(Tq):
            qs = i * Bq
            qe = min(qs + Bq, Nq)
            bq = qe - qs
            Qi = Qf[:, qs:qe, :].to(compute_dtype)

            Oi = torch.zeros(B, bq, d, device=Q.device, dtype=compute_dtype)
            li = torch.zeros(B, bq, device=Q.device, dtype=compute_dtype)
            mi = torch.full((B, bq), float("-inf"), device=Q.device, dtype=compute_dtype)

            for j in range(Tk):
                ks = j * Bk
                ke = min(ks + Bk, Nk)
                Kj = Kf[:, ks:ke, :].to(compute_dtype)
                Vj = Vf[:, ks:ke, :].to(compute_dtype)

                Sij = torch.matmul(Qi, Kj.transpose(-1, -2)) * scale  # (B, bq, bk)

                if is_causal:
                    q_idx = torch.arange(qs, qe, device=Q.device).unsqueeze(-1)
                    k_idx = torch.arange(ks, ke, device=Q.device).unsqueeze(0)
                    Sij = Sij + torch.where(q_idx >= k_idx, 0.0, -1e6)

                m_new = torch.maximum(mi, Sij.amax(dim=-1))
                P = torch.exp(Sij - m_new.unsqueeze(-1))
                alpha = torch.exp(mi - m_new)
                l_new = alpha * li + P.sum(dim=-1)
                Oi = alpha.unsqueeze(-1) * Oi + torch.matmul(P, Vj)

                mi = m_new
                li = l_new

            Oi = Oi / li.unsqueeze(-1)
            Li = mi + torch.log(li)

            O[:, qs:qe, :] = Oi.to(out_dtype)
            L[:, qs:qe] = Li

        O = O.reshape(*batch_dims, Nq, d)
        L = L.reshape(*batch_dims, Nq)

        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO: torch.Tensor):
        L, Q, K, V, O = ctx.saved_tensors
        dQ, dK, dV = _flash_backward_compiled(Q, K, V, O, dO, L, ctx.is_causal)
        return dQ, dK, dV, None


def _flash_backward_core(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
    O: torch.Tensor, dO: torch.Tensor, L: torch.Tensor,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FlashAttention-2 backward with recomputation (Eq. 13–19)."""
    d = Q.shape[-1]
    scale = 1.0 / math.sqrt(d)

    Qf = Q.to(torch.float32)
    Kf = K.to(torch.float32)
    Vf = V.to(torch.float32)
    Of = O.to(torch.float32)
    dOf = dO.to(torch.float32)
    Lf = L.to(torch.float32)

    D = (dOf * Of).sum(dim=-1)                                           # (..., Nq)
    S = torch.matmul(Qf, Kf.transpose(-1, -2)) * scale                   # (..., Nq, Nk)

    if is_causal:
        Nq, Nk = Qf.shape[-2], Kf.shape[-2]
        q_idx = torch.arange(Nq, device=Q.device).unsqueeze(-1)
        k_idx = torch.arange(Nk, device=Q.device).unsqueeze(0)
        S = S + torch.where(q_idx >= k_idx, 0.0, -1e6)

    P = torch.exp(S - Lf.unsqueeze(-1))                                  # Eq. 14
    dV = torch.matmul(P.transpose(-1, -2), dOf)                          # Eq. 15
    dP = torch.matmul(dOf, Vf.transpose(-1, -2))                         # Eq. 16
    dS = P * (dP - D.unsqueeze(-1))                                      # Eq. 17
    dQ = torch.matmul(dS, Kf) * scale                                    # Eq. 18
    dK = torch.matmul(dS.transpose(-1, -2), Qf) * scale                  # Eq. 19

    return dQ.to(Q.dtype), dK.to(K.dtype), dV.to(V.dtype)


_flash_backward_compiled = torch.compile(_flash_backward_core)
