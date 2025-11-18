from __future__ import annotations
import math
import torch
from torch import nn
from dataclasses import dataclass

@dataclass
class LMConfig:
    vocab_size: int
    n_embd: int = 384
    n_layer: int = 6
    n_head: int = 6
    max_seq_len: int = 512
    dropout: float = 0.1
    pad_id: int = 0

class CausalTransformerLM(nn.Module):
    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.n_embd)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.n_embd,
            nhead=cfg.n_head,
            dim_feedforward=cfg.n_embd * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layer)
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        self.register_buffer("mask", torch.triu(torch.ones(cfg.max_seq_len, cfg.max_seq_len), diagonal=1).bool())

    def forward(self, input_ids, attention_mask=None, labels=None):
        B, T = input_ids.shape
        device = input_ids.device
        pos = torch.arange(0, T, device=device).unsqueeze(0).expand(B, T)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        # Causal mask
        attn_mask = self.mask[:T, :T]
        x = self.blocks(x, mask=attn_mask)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if labels is not None:
            # shift for causal LM
            logits_shifted = logits[:, :-1, :].contiguous()
            labels_shifted = labels[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                logits_shifted.view(-1, logits_shifted.size(-1)),
                labels_shifted.view(-1),
                ignore_index=self.cfg.pad_id,
            )
        return {"loss": loss, "logits": logits}