import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
tokenizer.pad_token = tokenizer.eos_token  # unused in this stream-based setup


class MyGPTDataset(torch.utils.data.Dataset):
    """
    stream: 1D token-id sequence (list/np array/torch tensor), e.g. concatenated docs with EOS between them.
    Returns fixed-length chunks of token ids.
    """

    def __init__(self, stream, block_size=256):
        super().__init__()
        self.stream = torch.as_tensor(stream, dtype=torch.long)  # [N]
        self.block_size = int(block_size)

    def __getitem__(self, i):
        # return block_size+1 so you can do x=ids[:, :-1], y=ids[:, 1:]
        start = i * self.block_size
        end = start + self.block_size + 1
        return self.stream[start:end]  # [block_size+1]

    def __len__(self):
        # need +1 token for the shift
        return (len(self.stream) - 1) // self.block_size


def gpt_collate_fn(batch):
    # batch is list of 1D Long tensors with same length
    src_ids = torch.stack(batch, dim=0)  # [B, block_size+1]
    return src_ids
