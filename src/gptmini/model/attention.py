import math

import torch
import torch.nn.functional as F
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
    """
    query: (*B, S1, D)
    key:   (*B, S2, D)
    value: (*B, S2, D)
    returns: (*B, S1, D)
    """
    assert query.dim() >= 3, "expected at least 3D: (*B, S, D)"
    device = query.device
    dtype = query.dtype

    *batch_shape, S1, D = query.shape
    S2 = key.shape[-2]
    scale = 1.0 / math.sqrt(D)

    # Flatten batch dims -> N
    N = 1
    for b in batch_shape:
        N *= b

    q = query.reshape(N, S1, D)
    k = key.reshape(N, S2, D)
    v = value.reshape(N, S2, D)

    # Build masks in 2D and flatten per-batch if provided
    if is_causal:
        row = torch.arange(S1, device=device).unsqueeze(-1)
        col = torch.arange(S2, device=device).unsqueeze(0)
        causal_2d = row >= col  # (S1, S2)
    else:
        causal_2d = None

    if attn_mask is not None:
        # broadcast to (*B, S1, S2) then flatten to (N, S1, S2)
        attn_mask = torch.broadcast_to(attn_mask, (*batch_shape, S1, S2))
        attn_mask_flat = attn_mask.reshape(N, S1, S2).to(device=device)
    else:
        attn_mask_flat = None

    outs = []
    for i in range(N):
        # scores: (S1, S2) using 2D mm (works on your system)
        scores = torch.mm(q[i], k[i].transpose(0, 1)) * scale

        # combine masks
        if causal_2d is not None:
            mask_i = causal_2d
            if attn_mask_flat is not None:
                mask_i = mask_i & attn_mask_flat[i]
        else:
            mask_i = attn_mask_flat[i] if attn_mask_flat is not None else None

        if mask_i is not None:
            scores = scores.masked_fill(~mask_i, float("-inf"))

        probs = F.softmax(scores, dim=-1)

        if dropout:
            probs = F.dropout(probs, p=dropout, training=True)

        # out: (S1, D) using 2D mm
        out_i = torch.mm(probs, v[i])
        outs.append(out_i)

    out = torch.stack(outs, dim=0).reshape(*batch_shape, S1, D).to(dtype=dtype)
    return out


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
        return_cache=False,
    ):
        """

        :param data: Tensor of shape B, S1, input_dim
        :param context: Tensor of shape B, S2, input_dim
        :param pe: positional embedding module, should take Q and K as input and return transformed Q and K
        :param attn_mask: optional attention mask of shape B, S1, S2 or S1, S2. Masked positions should be False, unmasked - True
        :param kv_cache: (past_k, past_v): 2, B, H, T, head_dim
        :param return_cache: whether to return the new kv_cache (K, V) for future use. If False, cache will not be returned and not used.

        :return: Z, optional - kv_cache
        """
        B, S1 = data.shape[0], data.shape[-2]
        S2 = context.shape[-2]
        H = self.n_heads

        # Note: during generation S1 and S2 are usually 1 (or some small value), but during training they can be larger
        Q = self.q_projection(data)
        K = self.k_projection(context)
        V = self.v_projection(context)

        # print(Q.shape, B, S1, H, self.head_dim)
        Q = Q.view(B, S1, H, self.head_dim).transpose(1, 2)

        K = K.view(B, S2, H, self.head_dim).transpose(1, 2)
        V = V.view(B, S2, H, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            q_offset = kv_cache[0].shape[2]  # T
            k_offset = kv_cache[0].shape[2]  # T
        else:
            q_offset = 0
            k_offset = 0

        # rotary embeddings
        if pe is not None:
            Q, K = pe(Q, K, q_offset=q_offset, k_offset=k_offset)

        if kv_cache is not None:
            k_cache = kv_cache[0]  # B, H, T, head_dim
            v_cache = kv_cache[1]  # B, H, T, head_dim

            K = torch.cat([k_cache, K], dim=2)
            V = torch.cat([v_cache, V], dim=2)

        masking = Q.size(2) > 1

        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()

        Z = (
            scaled_dot_product_attention(
                Q,
                K,
                V,
                is_causal=self.is_causal & masking,
                dropout=self.dropout if self.training else 0.0,
                attn_mask=attn_mask,
            )
            .transpose(1, 2)
            .contiguous()
        )  # B, S1, H, d

        Z = Z.view(B, S1, -1)
        Z = self.output_projection(Z)
        if return_cache:
            return Z, (K, V)
        else:
            return Z
