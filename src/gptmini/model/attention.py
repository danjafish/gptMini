import math

import torch
from torch import nn


# B, S, emd_size
def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask=None,
    dropout=0.0,
    is_causal=False,
):
    scale = math.sqrt(query.size(-1))
    device = query.device
    # compute scores
    Z = torch.matmul(query, key.transpose(-2, -1))  # ...,S1 X ...,S2
    Z = Z / scale

    # masking
    if is_causal:
        row = torch.arange(Z.shape[-2], device=device).unsqueeze(-1)  # S1, 1
        col = torch.arange(Z.shape[-1], device=device).unsqueeze(0)  # 1, S2
        causal_mask = row >= col  # S1, S2
        causal_mask = causal_mask.broadcast_to(Z.shape)
    else:
        causal_mask = torch.ones(Z.shape, dtype=torch.bool, device=device)

    if attn_mask is not None:
        attn_mask = torch.broadcast_to(attn_mask, Z.shape)
    else:
        attn_mask = torch.ones(Z.shape, dtype=torch.bool, device=device)

    attn_mask = attn_mask & causal_mask
    attn_mask = attn_mask.to(device=device)

    Z = Z.masked_fill(~attn_mask, float("-inf"))

    Z = torch.nn.functional.softmax(Z, dim=-1)

    if dropout:
        Z = torch.nn.functional.dropout(Z, dropout, training=True)

    # weight values
    Z = torch.matmul(Z, value)

    return Z


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, feature_dim, n_heads, is_causal=False, dropout=0.0):
        super().__init__()

        self.n_heads = n_heads
        self.is_causal = is_causal
        assert feature_dim % n_heads == 0, "Feature dim should be dividable by n_heads"
        self.head_dim = feature_dim // n_heads

        self.q_projection = nn.Linear(input_dim, feature_dim)
        self.k_projection = nn.Linear(input_dim, feature_dim)
        self.v_projection = nn.Linear(input_dim, feature_dim)

        self.dropout = dropout
        self.output_projection = nn.Linear(feature_dim, input_dim)

    def forward(self, data: torch.Tensor, context: torch.Tensor, pe: nn.Module = None, attn_mask=None):
        B, S1 = data.shape[0], data.shape[-2]
        S2 = context.shape[-2]

        Q = self.q_projection(data)

        K = self.k_projection(context)
        V = self.v_projection(context)
        H = self.n_heads

        Q = Q.view(B, S1, H, self.head_dim).transpose(1, 2)

        K = K.view(B, S2, H, self.head_dim).transpose(1, 2)
        V = V.view(B, S2, H, self.head_dim).transpose(1, 2)

        # rotary embeddings
        if pe is not None:
            Q, K = pe(Q, K)

        Z = (
            scaled_dot_product_attention(
                Q,
                K,
                V,
                is_causal=self.is_causal,
                dropout=self.dropout if self.training else 0.0,
                attn_mask=attn_mask,
            )
            .transpose(1, 2)
            .contiguous()
        )  # B, S1, H, d

        Z = Z.view(B, S1, -1)
        Z = self.output_projection(Z)

        return Z
