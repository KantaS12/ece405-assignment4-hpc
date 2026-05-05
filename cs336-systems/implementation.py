import math

import torch
import torch.cuda.nvtx as nvtx
from einops import rearrange
from einops import einsum
import einx
from jaxtyping import Bool, Float, Int
from torch import Tensor
import torch.nn as nn

from cs336_basics.nn_utils import softmax

# Model

@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """Scaled dot-product attention.

    This function implements Eq. 1 of the Transformer paper.

    Args:
        Q: Tensor of queries, may have any number of leading dimensions.
        K: Tensor of keys, sharing leading dimensions with Q.
        V: Tensor of values, sharding leading dimensions with Q and K.
        mask: An (optional) mask of shape (..., seq_len, seq_len).
            Attention scores for positions with a mask value of `False` should
            be masked out, i.e., not affect the softmaxed attention probabilities.

    Returns:
        torch.FloatTensor of shape (..., seq_len, value_dimension)
        with the output of running your scaled dot product attention
        implementation with the provided key, query, and value tensors.
    """

    d_k = K.shape[-1]

    with nvtx.range("computing attention scores:"):
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

    if mask is not None:
        attention_scores = torch.where(mask, attention_scores, float("-inf"))

    with nvtx.range("computing softmax"):
        attention_weights = softmax(attention_scores, dim=-1)  # Softmax over the key dimension

    with nvtx.range("final matmul"):
        return einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")


@nvtx.range("casual multi-headed self attention forward")
def annotated_forward(self, x: Float[Tensor, " ... seq d_k"], token_positions: Int[Tensor, " ... seq"] | None = None) -> Float[Tensor, " ... seq d_v"]:
    """
    Args:
        x: The input to perform multi-headed self-attention on.
        positional_ids: The positional indices along the sequence dimension of the input embeddings.

    Returns:
        Self-attention outputs.
    """
    *b, sequence_length, d_model = x.size()
    assert d_model == self.d_model

    Q = self.q_proj(x)
    K = self.k_proj(x)
    V = self.v_proj(x)

    # Take apart each head from the embedding dimension of Q, K, V to shape (..., num_heads, seq_len, d_k).
    Q, K, V = (
        rearrange(X, "... seq (heads d) -> ... heads seq d", heads=self.num_heads)
        for X in (Q, K, V)
    )  # fmt: skip

    if token_positions is None:
        token_positions = einx.rearrange("seq -> b... seq", torch.arange(sequence_length, device=x.device), b=[1] * len(b))

    # Duplicate token positions for each head
    token_positions = rearrange(token_positions, "... seq -> ... 1 seq")

    with nvtx.range("computing positional encoding"):
        Q = self.positional_encoder(Q, token_positions)
        K = self.positional_encoder(K, token_positions)

    # Construct causal mask
    seq = torch.arange(sequence_length, device=x.device)

    with nvtx.range("constructing causal mask"):
        qi = einx.rearrange('query -> b... 1 query 1', seq, b=[1] * len(b))
        kj = einx.rearrange('key   -> b... 1 1   key', seq, b=[1] * len(b))
        causal_mask = qi >= kj  # (query, key)

    attn_output = annotated_scaled_dot_product_attention(K=K, Q=Q, V=V, mask=causal_mask)

    # Concatenate the attention output from all heads.
    # (..., sequence_length, num_heads * d_v).
    attn_output = rearrange(attn_output, "batch heads seq d_v -> batch seq (heads d_v)").contiguous()

    with nvtx.range("final matmul"):
        # Apply the output projection
        output = self.output_proj(attn_output)

    return output


def mixed_precision_accumulation():
    s = torch.tensor(0, dtype=torch.float32)
    for i in range(1000):
        s += torch.tensor(0.01,dtype=torch.float32)
    print(s)

    s = torch.tensor(0,dtype=torch.float16)
    for i in range(1000):
        s += torch.tensor(0.01,dtype=torch.float16)
    print(s)

    s = torch.tensor(0,dtype=torch.float32)
    for i in range(1000):
        s += torch.tensor(0.01,dtype=torch.float16)
    print(s)

    s = torch.tensor(0,dtype=torch.float32)
    for i in range(1000):
        x = torch.tensor(0.01,dtype=torch.float16)
        s += x.type(torch.float32)
    print(s)


# Benchmarking Mixed Precision
# Example

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x

# Model Parameters (FP 32). Autocast doesn't change parameter storage.
# fc1(x) output (FP 16). Matmul is autocasted to FP 16.
# LayerNorm (FP 32). On the FP32 list. 
# fc2(x) output (FP 16). Matmul is autocasted to FP 16.
# Loss (FP 32). On the FP32 list. Cross_entropy/ log_softmax / mse_loss.
# Parameter gradients (FP 32). On the FP32 list. Gradients are always stored in the same dtype as the parameters.
# Activation gradients during backward pass (FP 16). Through LN it is FP32 but through matmul it is FP16.
# https://pytorch.org/docs/stable/amp.html#cuda-ops-that-can-autocast-to-float16

# b
# Layer Norm
# To find the variance we need to do |x - mean|^2 which overflows FP16 if x is large. 
# Variance could also underflow if x is small.
# If X is around the mean, it could do a cancelation which could be really bad.
# Does B16 need to get special treatment?
# No, because B16 has the same range as FP32. The overflow and the underflow goes away which is why we use BF16 for training.
# However, the trade off is the less precision since it only has 7 bits instead of 10.
# Most modern architectures use BF16 but PyTorch keeps LayerNorm on FP32 for B16 too but it depends on the precision.

