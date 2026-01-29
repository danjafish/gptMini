import torch
import torch.nn.functional as F
from torch import nn


class TransLoss(nn.Module):
    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        assert 0.0 <= label_smoothing < 1.0
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, tgt_ids: torch.Tensor, pad_id: int):
        """
        logits: [B, T-1, V]  (predicts tokens 1..T-1)
        tgt_ids: [B, T]      (targets tokens 0..T-1)
        """
        B, S, V = logits.shape
        tgt_ids_shifted = tgt_ids[:, 1:]  # [B, T-1]

        return F.cross_entropy(
            logits.reshape(-1, V),
            tgt_ids_shifted.reshape(-1),
            ignore_index=pad_id,
            label_smoothing=self.label_smoothing,
        )


def train_step(
    model: nn.Module,
    batch: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    pad_id,
    scheduler=None,
):
    model.train()
    src_ids = batch

    src_ids = src_ids.to("cuda")

    logits = model(src_ids[:, :-1])

    loss = loss_fn(logits, src_ids, pad_id)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if scheduler:
        scheduler.step()

    return loss.item()


def eval_step(model: nn.Module, batch: torch.Tensor, loss_fn, pad_id):
    with torch.no_grad():
        model.eval()
        src_ids = batch
        src_ids = src_ids.to("cuda")

        logits = model(src_ids[:, :-1])

        loss = loss_fn(logits, src_ids, pad_id)

    return loss.item()
