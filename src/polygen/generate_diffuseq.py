"""
Generation script for DiffuSeq model.
"""
from __future__ import annotations
import argparse
import os
import torch
from .tokenizer import BigSMILESTokenizer
from .diffuseq_model import DiffuSeqModel, DiffuSeqConfig


def parse_args():
    ap = argparse.ArgumentParser(description="Generate sequences with DiffuSeq")
    ap.add_argument("--model-path", required=True, help="Path to trained model checkpoint")
    ap.add_argument("--tokenizer-dir", required=True, help="Path to tokenizer directory")
    ap.add_argument("--output", required=True, help="Output file for generated sequences")
    ap.add_argument("--num-samples", type=int, default=100, help="Number of sequences to generate")
    ap.add_argument("--seq-len", type=int, default=128, help="Length of generated sequences")
    ap.add_argument("--batch-size", type=int, default=16, help="Generation batch size")
    ap.add_argument("--num-steps", type=int, default=None, help="Number of denoising steps (default: use full schedule)")
    ap.add_argument("--device", default=None, help="Device (cuda/cpu, default: auto-detect)")
    return ap.parse_args()


def main():
    args = parse_args()
    
    # Load tokenizer
    print("Loading tokenizer...")
    tk = BigSMILESTokenizer.load(os.path.join(args.tokenizer_dir, "tokenizer.json"))
    
    # Set device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    checkpoint = torch.load(args.model_path, map_location=device)
    cfg_dict = checkpoint["cfg"]
    
    # Create config
    cfg = DiffuSeqConfig(**cfg_dict)
    
    # Create model and load weights
    model = DiffuSeqModel(cfg).to(device)
    model.schedule.to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    
    print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Generating {args.num_samples} sequences of length {args.seq_len}")
    
    # Generate sequences in batches
    all_sequences = []
    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    
    with torch.no_grad():
        for i in range(num_batches):
            batch_size = min(args.batch_size, args.num_samples - i * args.batch_size)
            
            print(f"Generating batch {i+1}/{num_batches} (size={batch_size})...")
            
            # Generate
            token_ids = model.generate(
                batch_size=batch_size,
                seq_len=args.seq_len,
                device=device,
                num_steps=args.num_steps,
            )
            
            # Decode
            for j in range(batch_size):
                ids = token_ids[j].cpu().tolist()
                # Remove special tokens
                ids = [id for id in ids if id not in [tk.pad_id(), tk.bos_id(), tk.eos_id()]]
                sequence = tk.decode(ids)
                all_sequences.append(sequence)
    
    # Save to file
    print(f"Saving {len(all_sequences)} sequences to {args.output}...")
    with open(args.output, "w") as f:
        for seq in all_sequences:
            f.write(seq + "\n")
    
    print("Generation complete!")
    print(f"Sample sequences:")
    for i, seq in enumerate(all_sequences[:5]):
        print(f"  {i+1}. {seq}")


if __name__ == "__main__":
    main()
