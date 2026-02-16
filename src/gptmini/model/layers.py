import torch
from torch import nn

from .attention import MultiHeadAttention


class PositionalEncoding(nn.Module):
    """
    Base class for *additive* positional encodings (e.g., learned absolute, sinusoidal/Fourier).
    Usage: x = pe(x) where x is (..., S, D).
    Subclasses must implement _compute(positions, device, dtype) -> (..., S, D) or (S, D).
    """

    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.0):
        super().__init__()
        self.d_model = int(d_model)
        self.max_len = int(max_len)
        self.dropout_p = float(dropout)
        self._dropout = nn.Dropout(self.dropout_p) if self.dropout_p > 0 else None

    def _compute(self, seq_len: int, device, dtype) -> torch.Tensor:
        """
        seq_len: integer.
        Return: (S,D) or (B,S,D) positional embeddings (same dtype/device as requested).
        """
        raise NotImplementedError

    @staticmethod
    def _default_positions(seq_len: int, device) -> torch.Tensor:
        return torch.arange(seq_len, device=device, dtype=torch.long)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x: B, S, ... - input tensor
        :return:  B, S, ... - output embeddings
        """
        return self._compute(x.size(1), x.device, x.dtype)


class FourierPositionalEncoding(PositionalEncoding):
    def __init__(self, d_model, max_len, dropout):
        super().__init__(d_model, max_len, dropout)
        self.base = 10000
        self.register_buffer("pre_emb", self._precompute_embeddings())

    def _precompute_embeddings(self) -> torch.Tensor:
        embeddings = torch.zeros(size=(self.max_len, self.d_model))

        pos = torch.arange(0, self.max_len).unsqueeze(-1)
        i = torch.arange(0, self.d_model, 2)

        div = torch.pow(torch.Tensor([self.base]), i / self.d_model).unsqueeze(0)

        angle = pos / div
        embeddings[:, 0::2] = torch.sin(angle)
        embeddings[:, 1::2] = torch.cos(angle)

        return embeddings

    def _compute(self, seq_len: int, device, dtype) -> torch.Tensor:
        if seq_len > self.max_len:
            raise RuntimeError(f"Positions will be clipped to max len = {self.max_len}")

        emb = self.pre_emb[:seq_len].to(device=device, dtype=dtype)
        return emb


class LearnedPositionalEncoding(PositionalEncoding):
    def __init__(self, d_model, max_len, dropout):
        super().__init__(d_model, max_len, dropout)
        self.embedding = nn.Embedding(max_len, d_model)

    def _compute(self, seq_len: int, device, dtype) -> torch.Tensor:
        return self.embedding(self._default_positions(seq_len, device))


class RoPositionalEncoding(nn.Module):
    def __init__(self, max_len, head_dim, base=10000):
        super().__init__()
        self.max_len = max_len
        self.base = base
        self.head_dim = head_dim

        sin, cos = self._precompute_rot_coef()
        self.register_buffer("cos", cos.view(1, 1, self.max_len, -1))
        self.register_buffer("sin", sin.view(1, 1, self.max_len, -1))

    def _precompute_rot_coef(self):
        pos = torch.arange(0, self.max_len).unsqueeze(-1)  # shape S,1

        i = torch.arange(0, self.head_dim // 2).unsqueeze(0)

        omega = torch.pow(self.base, -2 * i / self.head_dim)

        teta = pos * omega  # self.max_len, self.head_dim/2

        sin = torch.sin(teta)
        cos = torch.cos(teta)

        return sin, cos

    def apply_rot(self, x: torch.Tensor, sin, cos):
        x0 = x[..., 0]
        x1 = x[..., 1]

        y0 = cos * x0 - sin * x1
        y1 = sin * x0 + cos * x1

        x = torch.stack([y0, y1], dim=-1)
        return x

    def forward(self, query: torch.Tensor, key: torch.Tensor, q_offset=0, k_offset=0):
        B, H, S, D = query.size()
        _, H2, S2, D2 = key.size()

        assert H == H2 and D == D2
        assert D % 2 == 0

        sin_q = self.sin[:, :, q_offset : q_offset + S, :]
        cos_q = self.cos[:, :, q_offset : q_offset + S, :]
        sin_k = self.sin[:, :, k_offset : k_offset + S2, :]
        cos_k = self.cos[:, :, k_offset : k_offset + S2, :]

        sin_q, cos_q = sin_q.to(dtype=query.dtype), cos_q.to(dtype=query.dtype)
        sin_k, cos_k = sin_k.to(dtype=key.dtype), cos_k.to(dtype=key.dtype)

        q = self.apply_rot(query.view(B, H, S, D // 2, 2), sin_q, cos_q)
        k = self.apply_rot(key.view(B, H, S2, D2 // 2, 2), sin_k, cos_k)

        q = q.view(B, H, S, D)
        k = k.view(B, H2, S2, D2)

        return q, k


class FFN(nn.Module):
    def __init__(self, feature_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.hidden = nn.Linear(feature_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, feature_dim)

        self.dropout = dropout

    def forward(self, x):
        x = self.hidden(x)
        x = torch.nn.functional.gelu(x)
        if self.dropout:
            x = torch.nn.functional.dropout(x, self.dropout, training=self.training)
        x = self.output(x)
        if self.dropout:
            x = torch.nn.functional.dropout(x, self.dropout, training=self.training)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        n_heads,
        feature_dim,
        ffn_dim,
        ffn_dropout=0.0,
        attn_dropout=0.0,
    ):
        super().__init__()

        self.mha = MultiHeadAttention(
            input_dim, feature_dim, n_heads, is_causal=True, dropout=attn_dropout
        )

        self.ffn = FFN(input_dim, ffn_dim, ffn_dropout)

        self.pre_norm = nn.LayerNorm(input_dim)
        self.post_norm = nn.LayerNorm(input_dim)

    def forward(self, x, pe=None, attn_mask=None, past_kv=None, return_cache=False):
        attn_out = self.mha(
            self.pre_norm(x),
            self.pre_norm(x),
            pe,
            attn_mask,
            kv_cache=past_kv,
            return_cache=return_cache,
        )
        if return_cache:
            out, present_kv = attn_out
            x = x + out
        else:
            x = x + attn_out

        x = x + self.ffn(self.post_norm(x))

        if return_cache:
            return x, present_kv
        else:
            return x
