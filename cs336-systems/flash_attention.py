from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


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


@triton.jit
def _flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    Q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

    m_i = tl.full((Q_TILE_SIZE,), -1e6, dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    O_acc = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    q_offsets = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    n_key_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)

    for j in range(n_key_tiles):
        K = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        S = tl.dot(Q, tl.trans(K)).to(tl.float32) * scale  # (Bq, Bk)

        if IS_CAUSAL:
            k_offsets = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            S = tl.where(q_offsets[:, None] >= k_offsets[None, :], S, -1.0e6)

        m_new = tl.maximum(m_i, tl.max(S, axis=1))
        P = tl.exp(S - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = alpha * l_i + tl.sum(P, axis=1)
        O_acc = alpha[:, None] * O_acc + tl.dot(P.to(V.dtype), V).to(tl.float32)
        m_i = m_new

        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    O_acc = O_acc / l_i[:, None]
    L = m_i + tl.log(l_i)

    tl.store(O_block_ptr, O_acc.to(O_ptr.dtype.element_ty), boundary_check=(0, 1))
    tl.store(L_block_ptr, L, boundary_check=(0,))


class FlashAttentionTriton(torch.autograd.Function):
    Q_TILE_SIZE: int = 16
    K_TILE_SIZE: int = 16

    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        assert Q.is_cuda and K.is_cuda and V.is_cuda, "FlashAttentionTriton requires CUDA tensors"

        *batch_dims, Nq, D = Q.shape
        Nk = K.shape[-2]

        Qf = Q.reshape(-1, Nq, D).contiguous()
        Kf = K.reshape(-1, Nk, D).contiguous()
        Vf = V.reshape(-1, Nk, D).contiguous()
        B = Qf.shape[0]

        scale = 1.0 / math.sqrt(D)

        O = torch.empty((B, Nq, D), device=Q.device, dtype=Q.dtype)
        L = torch.empty((B, Nq), device=Q.device, dtype=torch.float32)

        Bq = FlashAttentionTriton.Q_TILE_SIZE
        Bk = FlashAttentionTriton.K_TILE_SIZE
        Tq = triton.cdiv(Nq, Bq)
        grid = (Tq, B)

        _flash_fwd_kernel[grid](
            Qf, Kf, Vf, O, L,
            Qf.stride(0), Qf.stride(1), Qf.stride(2),
            Kf.stride(0), Kf.stride(1), Kf.stride(2),
            Vf.stride(0), Vf.stride(1), Vf.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_QUERIES=Nq, N_KEYS=Nk,
            scale=scale,
            D=D,
            Q_TILE_SIZE=Bq, K_TILE_SIZE=Bk,
            IS_CAUSAL=is_causal,
        )

        O_out = O.reshape(*batch_dims, Nq, D)
        L_out = L.reshape(*batch_dims, Nq)

        ctx.save_for_backward(L_out, Q, K, V, O_out)
        ctx.is_causal = is_causal
        return O_out

    @staticmethod
    def backward(ctx, dO: torch.Tensor):
        L, Q, K, V, O = ctx.saved_tensors
        dQ, dK, dV = _flash_backward_compiled(Q, K, V, O, dO, L, ctx.is_causal)
        return dQ, dK, dV, None
