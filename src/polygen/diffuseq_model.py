"""
DiffuSeq: Discrete Diffusion Model for Token Sequence Generation

Based on the DiffuSeq paper, this implements a discrete diffusion process
for non-autoregressive sequence generation using token masking and denoising.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class DiffuSeqConfig:
    """Configuration for DiffuSeq model."""
    vocab_size: int
    n_embd: int = 384
    n_layer: int = 6
    n_head: int = 6
    max_seq_len: int = 512
    dropout: float = 0.1
    pad_id: int = 0
    mask_id: int = 3  # special token ID for [MASK]
    num_diffusion_steps: int = 2000
    schedule_type: str = "cosine"  # 'linear' or 'cosine'


class DiffusionSchedule:
    """Discrete diffusion schedule for token masking."""
    
    def __init__(self, num_steps: int, schedule_type: str = "cosine"):
        self.num_steps = num_steps
        self.schedule_type = schedule_type
        
        # Create diffusion schedule
        if schedule_type == "linear":
            self.betas = torch.linspace(0.0001, 0.02, num_steps)
        elif schedule_type == "cosine":
            # Cosine schedule: more gradual at start and end
            timesteps = torch.linspace(0, num_steps, num_steps + 1)
            alpha_bar = torch.cos(((timesteps / num_steps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            self.betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
            self.betas = torch.clamp(self.betas, 0.0001, 0.999)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
    def get_mask_prob(self, t: torch.Tensor) -> torch.Tensor:
        """Get probability of masking at timestep t."""
        # At t=0, no masking; at t=T, full masking
        return 1.0 - self.alphas_cumprod[t]
    
    def to(self, device):
        """Move schedule to device."""
        self.alphas = self.alphas.to(device)
        self.betas = self.betas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        return self


class DiffuSeqModel(nn.Module):
    """DiffuSeq denoising transformer model."""
    
    def __init__(self, cfg: DiffuSeqConfig):
        super().__init__()
        self.cfg = cfg
        
        # Token embeddings
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        
        # Positional embeddings
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.n_embd)
        
        # Time step embeddings for diffusion
        self.time_emb = nn.Sequential(
            nn.Linear(cfg.n_embd, cfg.n_embd * 4),
            nn.GELU(),
            nn.Linear(cfg.n_embd * 4, cfg.n_embd),
        )
        
        # Transformer encoder layers (bidirectional)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.n_embd,
            nhead=cfg.n_head,
            dim_feedforward=cfg.n_embd * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layer)
        
        # Final layer norm and output head
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        
        # Initialize diffusion schedule
        self.schedule = DiffusionSchedule(cfg.num_diffusion_steps, cfg.schedule_type)
        
    def get_time_embedding(self, timestep: torch.Tensor) -> torch.Tensor:
        """Create sinusoidal time embeddings."""
        device = timestep.device
        half_dim = self.cfg.n_embd // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = timestep[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.time_emb(emb)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        timestep: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the denoising model.
        
        Args:
            input_ids: [B, T] - noisy token IDs
            timestep: [B] - diffusion timestep
            attention_mask: [B, T] - attention mask (1 for valid, 0 for padding)
            
        Returns:
            logits: [B, T, V] - predicted token logits
        """
        B, T = input_ids.shape
        device = input_ids.device
        
        # Token embeddings
        tok_emb = self.tok_emb(input_ids)  # [B, T, D]
        
        # Position embeddings
        pos = torch.arange(0, T, device=device).unsqueeze(0).expand(B, T)
        pos_emb = self.pos_emb(pos)  # [B, T, D]
        
        # Time embeddings
        time_emb = self.get_time_embedding(timestep)  # [B, D]
        time_emb = time_emb.unsqueeze(1).expand(B, T, -1)  # [B, T, D]
        
        # Combine embeddings
        x = tok_emb + pos_emb + time_emb
        
        # Create attention mask for transformer (padding positions)
        if attention_mask is not None:
            # Convert to transformer format: 0 for valid, -inf for invalid
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None
        
        # Transformer (bidirectional, no causal mask)
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # Final layer norm and output
        x = self.ln_f(x)
        logits = self.head(x)  # [B, T, V]
        
        return logits
    
    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        mask_id: int,
    ) -> torch.Tensor:
        """
        Forward diffusion: add noise by masking tokens.
        
        Args:
            x_0: [B, T] - clean token IDs
            t: [B] - timestep for each sample
            mask_id: ID of the mask token
            
        Returns:
            x_t: [B, T] - noisy token IDs
        """
        B, T = x_0.shape
        device = x_0.device
        
        # Get masking probability for this timestep
        mask_prob = self.schedule.get_mask_prob(t)  # [B]
        
        # Sample which tokens to mask
        mask_prob_expanded = mask_prob[:, None].expand(B, T)  # [B, T]
        mask = torch.bernoulli(mask_prob_expanded).bool()  # [B, T]
        
        # Apply masking
        x_t = x_0.clone()
        x_t[mask] = mask_id
        
        return x_t
    
    def compute_loss(
        self,
        x_0: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute diffusion training loss.
        
        Args:
            x_0: [B, T] - clean token IDs
            attention_mask: [B, T] - attention mask
            
        Returns:
            loss: scalar tensor
        """
        B, T = x_0.shape
        device = x_0.device
        
        # Sample random timesteps
        t = torch.randint(0, self.cfg.num_diffusion_steps, (B,), device=device)
        
        # Forward diffusion: create noisy input
        x_t = self.q_sample(x_0, t, self.cfg.mask_id)
        
        # Predict original tokens from noisy input
        logits = self.forward(x_t, t, attention_mask)  # [B, T, V]
        
        # Compute loss only on masked positions
        mask = (x_t == self.cfg.mask_id)  # [B, T]
        
        # Apply attention mask if provided
        if attention_mask is not None:
            mask = mask & (attention_mask.bool())
        
        # Cross-entropy loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            x_0.view(-1),
            reduction='none',
        )
        loss = loss.view(B, T)
        
        # Average only over masked positions
        loss = (loss * mask.float()).sum() / (mask.sum() + 1e-8)
        
        return loss
    
    @torch.no_grad()
    def p_sample(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Reverse diffusion: denoise one step.
        
        Args:
            x_t: [B, T] - noisy tokens at timestep t
            t: [B] - current timestep
            attention_mask: [B, T] - attention mask
            
        Returns:
            x_t_minus_1: [B, T] - denoised tokens at timestep t-1
        """
        # Predict tokens
        logits = self.forward(x_t, t, attention_mask)  # [B, T, V]
        
        # Sample from predicted distribution
        probs = F.softmax(logits, dim=-1)
        x_pred = torch.argmax(probs, dim=-1)  # [B, T]
        
        # Only update masked positions
        mask = (x_t == self.cfg.mask_id)
        x_t_minus_1 = x_t.clone()
        x_t_minus_1[mask] = x_pred[mask]
        
        # Stochastic masking for next step (except at t=0)
        if t[0] > 0:
            # Remask some positions based on schedule
            next_mask_prob = self.schedule.get_mask_prob(t - 1)
            remask_prob = next_mask_prob / (self.schedule.get_mask_prob(t) + 1e-8)
            remask_prob = torch.clamp(remask_prob, 0, 1)
            
            remask = torch.bernoulli(
                remask_prob[:, None].expand_as(x_t_minus_1)
            ).bool() & mask
            x_t_minus_1[remask] = self.cfg.mask_id
        
        return x_t_minus_1
    
    @torch.no_grad()
    def generate(
        self,
        batch_size: int = 1,
        seq_len: int = 128,
        device: str = "cpu",
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate sequences using reverse diffusion.
        
        Args:
            batch_size: number of sequences to generate
            seq_len: length of sequences
            device: device to generate on
            num_steps: number of denoising steps (default: full schedule)
            
        Returns:
            x_0: [B, T] - generated token IDs
        """
        if num_steps is None:
            num_steps = self.cfg.num_diffusion_steps
        
        # Start with all masked tokens
        x_t = torch.full(
            (batch_size, seq_len),
            self.cfg.mask_id,
            dtype=torch.long,
            device=device,
        )
        
        # Reverse diffusion
        timesteps = list(range(num_steps))[::-1]  # [T-1, T-2, ..., 1, 0]
        
        for t in timesteps:
            t_batch = torch.full((batch_size,), t, dtype=torch.long, device=device)
            x_t = self.p_sample(x_t, t_batch)
        
        return x_t
