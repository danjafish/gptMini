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

    def forward(self, x: torch.Tensor, attn_mask=None):
        # embedding
        # print('Shape before embedding: ', x.shape)
        x = self.token_embedding(x)
        # print('Shape after embedding: ', x.shape)

        if self.emb_type in ["learned", "fourier"]:
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

    def genereate(self, prompt, max_steps=100, strategy="greedy", t=0.0, eos_token=0):
        """
        Generation function

        :param prompt: Tensor, shape B, S - tokens
        :param max_steps: max steps to generate
        :param strategy: strategy to use for generation
        :param t: temperature

        :return: B, M - generated sequence
        """
        self.eval()
        with torch.no_grad():
            B = prompt.size(0)
            is_finished = torch.zeros((B, 1), device=prompt.device, dtype=torch.bool)
            generated = prompt.clone()

            if strategy == "greedy":
                for step in range(max_steps):
                    logits = self.forward(generated)  # B, S, vocab_size

                    next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(
                        1
                    )  # pick last token, and argmax -> B, 1
                    is_finished = is_finished | (next_token == eos_token)

                    next_token[is_finished] = eos_token

                    generated = torch.cat([generated, next_token], dim=1)

                    if is_finished.all():
                        break
            else:
                raise RuntimeError("only greedy strategy is implemented for now")

            return generated
