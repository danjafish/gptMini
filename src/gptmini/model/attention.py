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

    def forward(
        self,
        data: torch.Tensor,
        context: torch.Tensor,
        pe: nn.Module = None,
        attn_mask=None,
        kv_cache=None,
    ):
        """

        :param data:
        :param context:
        :param pe:
        :param attn_mask:
        :param kv_cache: (past_k, past_v): 2, B, S2, H, head_dim
        :return: Z, optional - kv_cache
        """
        B, S1 = data.shape[0], data.shape[-2]
        S2 = context.shape[-2]
        H = self.n_heads

        Q = self.q_projection(data)
        if kv_cache is None:
            K = self.k_projection(context)
            V = self.v_projection(context)
        else:
            # we only need to compute one token and update K and V
            last_token = context[:, -1].view(B, 1, -1)
            k = self.k_projection(last_token)
            v = self.v_projection(last_token)

            if pe is not None:
                Q, K = pe(Q, K)

            k_cache = kv_cache[0].view(B, S2, -1)  # transfer to same shape as k/v
            v_cache = kv_cache[1].view(B, S2, -1)

            K = torch.cat([k_cache, k], dim=1)
            V = torch.cat([v_cache, v], dim=1)

        # print(Q.shape, B, S1, H, self.head_dim)
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

        return Z, (K, V)
