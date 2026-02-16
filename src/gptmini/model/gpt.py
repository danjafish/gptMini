import torch
from torch import nn
from torch.nn import functional as F

from .layers import (
    FourierPositionalEncoding,
    LearnedPositionalEncoding,
    RoPositionalEncoding,
    TransformerBlock,
)


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
        emb_type="rotary",
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
                )
                for _ in range(n_layers)
            ]
        )
        pe = {
            "rotary": RoPositionalEncoding(
                max_len=max_len, head_dim=input_dim // n_heads
            ),
            "fourier": FourierPositionalEncoding(
                d_model=input_dim, max_len=max_len, dropout=attn_dropout
            ),
            "learned": LearnedPositionalEncoding(
                d_model=input_dim, max_len=max_len, dropout=attn_dropout
            ),
        }
        self.emb_type = emb_type
        if self.emb_type not in pe:
            raise RuntimeError(f"unknown embedding type {self.emb_type}")
        else:
            self.pe = pe[self.emb_type]

        self.token_embedding = nn.Embedding(vocab_len, input_dim)

        self.gpt_head = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, vocab_len),
        )

    def forward(
        self, x: torch.Tensor, attn_mask=None, return_cache=False, past_kv_list=None
    ):
        # embedding
        # print('Shape before embedding: ', x.shape)
        x = self.token_embedding(x)
        # print('Shape after embedding: ', x.shape)

        if self.emb_type in ["learned", "fourier"]:
            pe_emb = self.pe(x)
            x = x + pe_emb
            pe = None
        else:
            pe = self.pe
        if return_cache and past_kv_list is None:
            past_kv_list = [None for _ in range(len(self.decoder_layers))]

        for i, layer in enumerate(self.decoder_layers):
            if return_cache:
                x, present_kv = layer(
                    x, pe, attn_mask, past_kv=past_kv_list[i], return_cache=return_cache
                )
                past_kv_list[i] = present_kv
            else:
                x = layer(x, pe, attn_mask)

        x = self.norm(x)
        out = self.gpt_head(x)
        if return_cache:
            return out, past_kv_list
        else:
            return out

    def generate(
        self, prompt, max_steps=100, strategy="greedy", t=1.0, top_k=50, eos_token=0
    ):
        self.eval()
        with torch.no_grad():
            B = prompt.size(0)
            is_finished = torch.zeros((B, 1), device=prompt.device, dtype=torch.bool)
            generated = prompt.clone()

            # prime cache with the whole prompt
            logits, present_kv_list = self.forward(
                generated, past_kv_list=None, return_cache=True
            )

            for step in range(max_steps):
                last_logits = logits[:, -1, :]  # (B, V)

                if strategy == "greedy":
                    next_token = torch.argmax(last_logits, dim=-1, keepdim=True)

                elif strategy == "topk":
                    temp = 1.0 if (t is None or t <= 0.0) else float(t)
                    last_logits = last_logits / temp

                    k = min(int(top_k), last_logits.size(-1))
                    topv, topi = torch.topk(last_logits, k, dim=-1)  # (B, k), (B, k)
                    probs = torch.softmax(topv, dim=-1)  # (B, k)
                    sample_idx = torch.multinomial(probs, num_samples=1)  # (B, 1)
                    next_token = topi.gather(-1, sample_idx)  # (B, 1)

                else:
                    raise RuntimeError("strategy must be 'greedy' or 'topk'")

                is_finished = is_finished | (next_token == eos_token)
                next_token[is_finished] = eos_token

                generated = torch.cat([generated, next_token], dim=1)
                if is_finished.all():
                    break

                logits, present_kv_list = self.forward(
                    next_token, past_kv_list=present_kv_list, return_cache=True
                )

            return generated
