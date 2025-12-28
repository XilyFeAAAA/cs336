from typing import Optional
from infra.softmax import Softmax
from infra.rope import RoPE
from infra.linear import Linear
import numpy as np
import torch.nn as nn
import torch


def scaled_dot_production_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
):
    *_, t_q, d_k = q.shape
    *_, t_k, d_k = k.shape

    scores = q @ k.transpose(-1, -2) / np.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask[:t_q, :t_k] == False, float("-inf"))

    attn = Softmax(scores, dim=-1)
    attn = attn @ v

    return attn


class Multihead_Self_Attention(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        use_rope: bool,
        device: Optional[torch.device],
        *,
        theta: float,
        d_k: int,
        max_seq_len: int
    ):
        super(Multihead_Self_Attention, self).__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_rope = use_rope
        self.mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool, device=device))

        if use_rope:
            self.rope = RoPE(theta, d_k, max_seq_len, device)

        self.q_proj = Linear(d_model, d_model, device=device)
        self.k_proj = Linear(d_model, d_model, device=device)
        self.v_proj = Linear(d_model, d_model, device=device)
        self.output_proj = Linear(d_model, d_model, device=device)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: Optional[torch.Tensor] = None,
    ):

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(*x.shape[:-1], self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(*x.shape[:-1], self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(*x.shape[:-1], self.num_heads, self.d_k).transpose(1, 2)

        if self.use_rope:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        attn = scaled_dot_production_attention(q, k, v, self.mask)
        attn = attn.transpose(1, 2).contiguous().view_as(x)
        return self.output_proj(attn)
