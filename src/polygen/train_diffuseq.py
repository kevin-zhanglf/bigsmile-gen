"""
Training script for DiffuSeq model.
"""
from __future__ import annotations
import argparse
import os
import yaml
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from .tokenizer import BigSMILESTokenizer
from .diffuseq_dataset import DiffuSeqDataset, collate_diffuseq_batch
from .diffuseq_model import DiffuSeqModel, DiffuSeqConfig


def parse_args():
    ap = argparse.ArgumentParser(description="Train DiffuSeq model")
    ap.add_argument("--dataset", required=True, help="Path to CSV dataset")
    ap.add_argument("--text-column", default="bigsmiles", help="Column name for text")
    ap.add_argument("--tokenizer-dir", required=True, help="Path to tokenizer directory")
    ap.add_argument("--out-dir", required=True, help="Output directory for checkpoints")
    ap.add_argument("--config", required=True, help="Path to config YAML")
    ap.add_argument("--epochs", type=int, default=None, help="Override epochs from config")
    ap.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    ap.add_argument("--lr", type=float, default=None, help="Override learning rate")
    ap.add_argument("--num-diffusion-steps", type=int, default=None, help="Number of diffusion steps")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load tokenizer
    tk = BigSMILESTokenizer.load(os.path.join(args.tokenizer_dir, "tokenizer.json"))
    
    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    
    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {})
    diffusion_cfg = cfg.get("diffusion", {})
    
    # Override from command line
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size
    if args.lr is not None:
        train_cfg["lr"] = args.lr
    if args.num_diffusion_steps is not None:
        diffusion_cfg["num_steps"] = args.num_diffusion_steps
    
    # Set defaults
    train_cfg.setdefault("epochs", 10)
    train_cfg.setdefault("batch_size", 16)
    train_cfg.setdefault("lr", 1e-4)
    train_cfg.setdefault("gradient_clip", 1.0)
    train_cfg.setdefault("log_interval", 50)
    train_cfg.setdefault("save_interval", 1)
    
    model_cfg.setdefault("n_embd", 384)
    model_cfg.setdefault("n_layer", 6)
    model_cfg.setdefault("n_head", 6)
    model_cfg.setdefault("max_seq_len", 512)
    model_cfg.setdefault("dropout", 0.1)
    
    diffusion_cfg.setdefault("num_steps", 2000)
    diffusion_cfg.setdefault("schedule_type", "cosine")
    
    # Create dataset and dataloader
    dataset = DiffuSeqDataset(
        args.dataset,
        args.text_column,
        tk,
        max_length=model_cfg["max_seq_len"],
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        collate_fn=lambda b: collate_diffuseq_batch(b, tk.pad_id()),
        num_workers=0,
    )
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create model
    diffuseq_cfg = DiffuSeqConfig(
        vocab_size=tk.vocab_size,
        n_embd=model_cfg["n_embd"],
        n_layer=model_cfg["n_layer"],
        n_head=model_cfg["n_head"],
        max_seq_len=model_cfg["max_seq_len"],
        dropout=model_cfg["dropout"],
        pad_id=tk.pad_id(),
        mask_id=tk.tk.token_to_id("<unk>"),  # Use <unk> as mask token
        num_diffusion_steps=diffusion_cfg["num_steps"],
        schedule_type=diffusion_cfg["schedule_type"],
    )
    
    model = DiffuSeqModel(diffuseq_cfg).to(device)
    model.schedule.to(device)
    
    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Training for {train_cfg['epochs']} epochs")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )
    
    # Training loop
    model.train()
    global_step = 0
    
    for epoch in range(train_cfg["epochs"]):
        epoch_loss = 0.0
        pbar = tqdm(
            dataloader,
            desc=f"Epoch {epoch+1}/{train_cfg['epochs']}",
        )
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            loss = model.compute_loss(
                batch["input_ids"],
                batch["attention_mask"],
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                train_cfg["gradient_clip"],
            )
            
            # Optimizer step
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            # Logging
            global_step += 1
            epoch_loss += loss.item()
            
            if batch_idx % train_cfg["log_interval"] == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Epoch summary
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % train_cfg["save_interval"] == 0:
            ckpt_path = os.path.join(args.out_dir, f"diffuseq_epoch{epoch+1}.pt")
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "cfg": diffuseq_cfg.__dict__,
                "epoch": epoch + 1,
                "global_step": global_step,
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")
    
    # Save final model
    final_path = os.path.join(args.out_dir, "diffuseq_final.pt")
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "cfg": diffuseq_cfg.__dict__,
        "epoch": train_cfg["epochs"],
        "global_step": global_step,
    }, final_path)
    print(f"Training complete! Final model saved: {final_path}")


if __name__ == "__main__":
    main()
