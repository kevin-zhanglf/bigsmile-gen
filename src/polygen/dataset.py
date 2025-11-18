from __future__ import annotations
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Optional, List, Dict
from .tokenizer import BigSMILESTokenizer

class BigSMILESDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        text_column: str,
        tokenizer: BigSMILESTokenizer,
        max_length: int = 512,
    ):
        self.df = pd.read_csv(csv_path)
        self.text_col = text_column
        self.tk = tokenizer
        self.max_length = max_length

        # simple cleaning: drop NaN
        self.df = self.df[self.df[self.text_col].notna()].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.df.loc[idx, self.text_col])
        ids = self.tk.encode(text)[: self.max_length]
        # create labels identical to input for LM
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(ids, dtype=torch.long),
        }

def collate_batch(batch: List[Dict[str, torch.Tensor]], pad_id: int):
    # left pad not required for causal LM; pad to same length
    max_len = max(item["input_ids"].shape[0] for item in batch)
    input_ids = []
    labels = []
    for item in batch:
        ids = item["input_ids"]
        pad_len = max_len - ids.shape[0]
        if pad_len > 0:
            pad = torch.full((pad_len,), pad_id, dtype=torch.long)
            ids_padded = torch.cat([ids, pad], dim=0)
        else:
            ids_padded = ids
        input_ids.append(ids_padded)
        labels.append(ids_padded.clone())
    return {
        "input_ids": torch.stack(input_ids, dim=0),
        "labels": torch.stack(labels, dim=0),
        "attention_mask": (torch.stack(input_ids, dim=0) != pad_id).long(),
    }