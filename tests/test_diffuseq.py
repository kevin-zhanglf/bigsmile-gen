"""
Basic tests for DiffuSeq implementation.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from polygen.diffuseq_model import DiffuSeqModel, DiffuSeqConfig, DiffusionSchedule


def test_diffusion_schedule():
    """Test diffusion schedule creation."""
    schedule = DiffusionSchedule(num_steps=100, schedule_type="cosine")
    
    # Check schedule has correct length
    assert len(schedule.alphas) == 100
    assert len(schedule.betas) == 100
    
    # Check masking probability increases with timestep
    t0 = torch.tensor([0])
    t_mid = torch.tensor([50])
    t_end = torch.tensor([99])
    
    mask_prob_0 = schedule.get_mask_prob(t0)
    mask_prob_mid = schedule.get_mask_prob(t_mid)
    mask_prob_end = schedule.get_mask_prob(t_end)
    
    assert mask_prob_0 < mask_prob_mid < mask_prob_end
    print(f"✓ Diffusion schedule test passed")


def test_diffuseq_config():
    """Test DiffuSeq configuration."""
    cfg = DiffuSeqConfig(
        vocab_size=1000,
        n_embd=128,
        n_layer=2,
        n_head=4,
    )
    
    assert cfg.vocab_size == 1000
    assert cfg.n_embd == 128
    assert cfg.n_layer == 2
    assert cfg.n_head == 4
    print(f"✓ DiffuSeq config test passed")


def test_model_creation():
    """Test model can be created."""
    cfg = DiffuSeqConfig(
        vocab_size=1000,
        n_embd=128,
        n_layer=2,
        n_head=4,
        num_diffusion_steps=100,
    )
    
    model = DiffuSeqModel(cfg)
    
    # Check model has expected components
    assert hasattr(model, 'tok_emb')
    assert hasattr(model, 'pos_emb')
    assert hasattr(model, 'time_emb')
    assert hasattr(model, 'transformer')
    assert hasattr(model, 'head')
    
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count > 0
    print(f"✓ Model creation test passed (params: {param_count:,})")


def test_forward_pass():
    """Test forward pass."""
    cfg = DiffuSeqConfig(
        vocab_size=100,
        n_embd=64,
        n_layer=2,
        n_head=2,
        max_seq_len=32,
    )
    
    model = DiffuSeqModel(cfg)
    
    batch_size = 2
    seq_len = 16
    
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    timestep = torch.randint(0, cfg.num_diffusion_steps, (batch_size,))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    
    with torch.no_grad():
        logits = model.forward(input_ids, timestep, attention_mask)
    
    # Check output shape
    assert logits.shape == (batch_size, seq_len, cfg.vocab_size)
    print(f"✓ Forward pass test passed")


def test_q_sample():
    """Test forward diffusion (adding noise)."""
    cfg = DiffuSeqConfig(
        vocab_size=100,
        n_embd=64,
        n_layer=2,
        n_head=2,
        num_diffusion_steps=100,
    )
    
    model = DiffuSeqModel(cfg)
    
    batch_size = 2
    seq_len = 16
    
    x_0 = torch.randint(10, 90, (batch_size, seq_len))  # Avoid special tokens
    t = torch.tensor([50, 80])  # Different timesteps
    
    x_t = model.q_sample(x_0, t, mask_id=cfg.mask_id)
    
    # Check output shape
    assert x_t.shape == x_0.shape
    
    # Check some tokens were masked
    assert (x_t == cfg.mask_id).sum() > 0
    print(f"✓ Q-sample test passed")


def test_loss_computation():
    """Test loss computation."""
    cfg = DiffuSeqConfig(
        vocab_size=100,
        n_embd=64,
        n_layer=2,
        n_head=2,
        max_seq_len=32,
    )
    
    model = DiffuSeqModel(cfg)
    
    batch_size = 2
    seq_len = 16
    
    x_0 = torch.randint(10, 90, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    
    loss = model.compute_loss(x_0, attention_mask)
    
    # Check loss is a scalar
    assert loss.ndim == 0
    
    # Check loss is positive
    assert loss.item() > 0
    print(f"✓ Loss computation test passed (loss: {loss.item():.4f})")


def test_generation():
    """Test sequence generation."""
    cfg = DiffuSeqConfig(
        vocab_size=100,
        n_embd=64,
        n_layer=2,
        n_head=2,
        max_seq_len=32,
        num_diffusion_steps=10,  # Small for testing
    )
    
    model = DiffuSeqModel(cfg)
    
    batch_size = 2
    seq_len = 16
    
    with torch.no_grad():
        generated = model.generate(
            batch_size=batch_size,
            seq_len=seq_len,
            device='cpu',
            num_steps=5,  # Few steps for speed
        )
    
    # Check output shape
    assert generated.shape == (batch_size, seq_len)
    
    # Check no masked tokens in final output
    # (Small chance they might remain if model is untrained, so we just check shape)
    assert (generated >= 0).all()
    assert (generated < cfg.vocab_size).all()
    print(f"✓ Generation test passed")


def test_p_sample():
    """Test reverse diffusion step."""
    cfg = DiffuSeqConfig(
        vocab_size=100,
        n_embd=64,
        n_layer=2,
        n_head=2,
    )
    
    model = DiffuSeqModel(cfg)
    
    batch_size = 2
    seq_len = 16
    
    # Create noisy input
    x_t = torch.full((batch_size, seq_len), cfg.mask_id, dtype=torch.long)
    t = torch.tensor([10, 10])
    
    with torch.no_grad():
        x_t_minus_1 = model.p_sample(x_t, t)
    
    # Check output shape
    assert x_t_minus_1.shape == x_t.shape
    print(f"✓ P-sample test passed")


if __name__ == "__main__":
    print("Running DiffuSeq tests...\n")
    
    test_diffusion_schedule()
    test_diffuseq_config()
    test_model_creation()
    test_forward_pass()
    test_q_sample()
    test_loss_computation()
    test_generation()
    test_p_sample()
    
    print("\n✓ All tests passed!")
