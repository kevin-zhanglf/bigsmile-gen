"""
Dataset and data collator for DiffuSeq training.
"""
from __future__ import annotations
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Dict
from .tokenizer import BigSMILESTokenizer


class DiffuSeqDataset(Dataset):
    """Dataset for DiffuSeq training."""
    
    def __init__(
        self,
        csv_path: str,
        text_column: str,
        tokenizer: BigSMILESTokenizer,
        max_length: int = 512,
    ):
        """
        Initialize DiffuSeq dataset.
        
        Args:
            csv_path: path to CSV file with text data
            text_column: name of column containing text
            tokenizer: BigSMILES tokenizer
            max_length: maximum sequence length
        """
        self.df = pd.read_csv(csv_path)
        self.text_col = text_column
        self.tk = tokenizer
        self.max_length = max_length
        
        # Clean: drop NaN
        self.df = self.df[self.df[self.text_col].notna()].reset_index(drop=True)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training example.
        
        Returns:
            dict with 'input_ids' - the clean token sequence
        """
        text = str(self.df.loc[idx, self.text_col])
        ids = self.tk.encode(text)[: self.max_length]
        
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
        }


def collate_diffuseq_batch(
    batch: List[Dict[str, torch.Tensor]],
    pad_id: int,
) -> Dict[str, torch.Tensor]:
    """
    Collate a batch for DiffuSeq training.
    
    Args:
        batch: list of examples from dataset
        pad_id: padding token ID
        
    Returns:
        dict with:
            - input_ids: [B, T] padded token sequences
            - attention_mask: [B, T] mask (1=valid, 0=padding)
    """
    # Find max length in batch
    max_len = max(item["input_ids"].shape[0] for item in batch)
    
    input_ids = []
    attention_masks = []
    
    for item in batch:
        ids = item["input_ids"]
        pad_len = max_len - ids.shape[0]
        
        if pad_len > 0:
            # Right pad
            pad = torch.full((pad_len,), pad_id, dtype=torch.long)
            ids_padded = torch.cat([ids, pad], dim=0)
            mask = torch.cat([torch.ones(ids.shape[0]), torch.zeros(pad_len)], dim=0)
        else:
            ids_padded = ids
            mask = torch.ones(ids.shape[0])
        
        input_ids.append(ids_padded)
        attention_masks.append(mask)
    
    return {
        "input_ids": torch.stack(input_ids, dim=0),
        "attention_mask": torch.stack(attention_masks, dim=0).long(),
    }
