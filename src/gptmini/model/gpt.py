import torch
from torch import nn
from layers import TransformerBlock, FourierPositionalEncoding, RoPositionalEncoding, LearnedPositionalEncoding


class GPTmini(nn.Module):
    def __init__(
        self,
        max_len,
        vocab_len,
        input_dim,
        n_layers,
        n_heads,
        feature_dim,
        ffn_dim,
        ffn_dropout=0.0,
        attn_dropout=0.0,
        emb_type='rotary',
    ):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.decoder_layers = nn.ModuleList(
            [
                TransformerBlock(
                    input_dim,
                    n_heads,
                    feature_dim,
                    ffn_dim,
                    ffn_dropout=ffn_dropout,
                    attn_dropout=attn_dropout,
                ) for _ in range(n_layers)
            ]
        )
        pe = {
            'rotary': RoPositionalEncoding(max_len=max_len, head_dim=input_dim // n_heads),
            'fourier': FourierPositionalEncoding(d_model=input_dim, max_len=max_len, dropout=attn_dropout),
            'learned': LearnedPositionalEncoding(d_model=input_dim, max_len=max_len, dropout=attn_dropout),
        }
        self.emb_type = emb_type
        if self.emb_type not in pe:
            raise RuntimeError(f'unknown embedding type {self.emb_type}')
        else:
            self.pe = pe[self.emb_type]

        self.token_embedding = nn.Embedding(vocab_len, input_dim)

        self.gpt_head = nn.Sequential(
            nn.Linear(input_dim, input_dim*2),
            nn.GELU(),
            nn.Linear(input_dim*2, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, vocab_len)
        )

    def forward(self, x: torch.Tensor, attn_mask=None):
        # embedding
        x = self.token_embedding(x)

        if self.emb_type in ['learned', 'fourier']:
            pe_emb = self.pe(x, x.size(1))
            x = x + pe_emb
            pe = None
        else:
            pe = self.pe

        for layer in self.decoder_layers:
            x = layer(x, pe, attn_mask)

        x = self.norm(x)
        out = self.gpt_head(x)
        return out
