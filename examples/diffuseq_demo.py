"""
Demo script showing how to use DiffuSeq programmatically.

This demonstrates:
1. Creating a DiffuSeq model
2. Training on sample data
3. Generating sequences
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from polygen.diffuseq_model import DiffuSeqModel, DiffuSeqConfig


def demo_diffuseq():
    """Demonstrate DiffuSeq usage."""
    print("=" * 60)
    print("DiffuSeq Demo - Discrete Diffusion for Sequence Generation")
    print("=" * 60)
    print()
    
    # 1. Create model configuration
    print("1. Creating DiffuSeq model...")
    cfg = DiffuSeqConfig(
        vocab_size=100,        # Small vocabulary for demo
        n_embd=128,            # Embedding dimension
        n_layer=4,             # Number of transformer layers
        n_head=4,              # Number of attention heads
        max_seq_len=64,        # Maximum sequence length
        dropout=0.1,
        pad_id=0,
        mask_id=3,
        num_diffusion_steps=100,  # Number of diffusion timesteps
        schedule_type="cosine",   # Cosine noise schedule
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Using device: {device}")
    
    model = DiffuSeqModel(cfg).to(device)
    model.schedule.to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {param_count:,}")
    print()
    
    # 2. Simulate training on dummy data
    print("2. Simulating training (on dummy data)...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    num_iterations = 10
    batch_size = 4
    seq_len = 32
    
    for i in range(num_iterations):
        # Generate random training data
        x_0 = torch.randint(4, cfg.vocab_size, (batch_size, seq_len), device=device)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
        
        # Compute loss
        loss = model.compute_loss(x_0, attention_mask)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if (i + 1) % 5 == 0:
            print(f"   Iteration {i+1}/{num_iterations} - Loss: {loss.item():.4f}")
    
    print()
    
    # 3. Generate sequences
    print("3. Generating sequences...")
    model.eval()
    
    with torch.no_grad():
        # Generate with full diffusion process
        generated_full = model.generate(
            batch_size=3,
            seq_len=32,
            device=device,
            num_steps=100,  # Full denoising
        )
        
        # Generate with accelerated denoising
        generated_fast = model.generate(
            batch_size=3,
            seq_len=32,
            device=device,
            num_steps=20,  # Faster, fewer steps
        )
    
    print(f"   Generated with full denoising (100 steps):")
    for i, seq in enumerate(generated_full):
        print(f"     Seq {i+1}: {seq[:16].cpu().tolist()}... (showing first 16 tokens)")
    
    print()
    print(f"   Generated with accelerated denoising (20 steps):")
    for i, seq in enumerate(generated_fast):
        print(f"     Seq {i+1}: {seq[:16].cpu().tolist()}... (showing first 16 tokens)")
    
    print()
    
    # 4. Demonstrate diffusion process
    print("4. Demonstrating diffusion process...")
    
    # Original sequence
    x_0 = torch.randint(10, 90, (1, 16), device=device)
    print(f"   Original sequence: {x_0[0].cpu().tolist()}")
    
    # Forward diffusion at different timesteps
    for t_val in [10, 30, 50, 70, 90]:
        t = torch.tensor([t_val], device=device)
        x_t = model.q_sample(x_0, t, mask_id=cfg.mask_id)
        num_masked = (x_t == cfg.mask_id).sum().item()
        print(f"   After {t_val} steps: {num_masked}/16 tokens masked")
    
    print()
    
    # 5. Show model components
    print("5. Model architecture summary:")
    print(f"   - Token embedding: {cfg.vocab_size} x {cfg.n_embd}")
    print(f"   - Position embedding: {cfg.max_seq_len} x {cfg.n_embd}")
    print(f"   - Time embedding: MLP projecting to {cfg.n_embd}")
    print(f"   - Transformer: {cfg.n_layer} layers, {cfg.n_head} heads")
    print(f"   - Output head: {cfg.n_embd} → {cfg.vocab_size}")
    print(f"   - Diffusion schedule: {cfg.schedule_type}, {cfg.num_diffusion_steps} steps")
    print()
    
    print("=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Run demo
    demo_diffuseq()
    
    print()
    print("Key takeaways:")
    print("- DiffuSeq uses discrete diffusion to generate sequences")
    print("- Forward process: gradually mask tokens")
    print("- Reverse process: iteratively denoise to generate")
    print("- Non-autoregressive: all positions decoded in parallel")
    print("- Trade-off: more steps = better quality, fewer steps = faster")
