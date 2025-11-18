from __future__ import annotations
import argparse
import os
import yaml
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from .tokenizer import BigSMILESTokenizer
from .dataset import BigSMILESDataset, collate_batch
from .modeling import CausalTransformerLM, LMConfig

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--text-column", default="bigsmiles")
    ap.add_argument("--tokenizer-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    tk = BigSMILESTokenizer.load(os.path.join(args.tokenizer_dir, "tokenizer.json"))

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]

    if args.epochs is not None: train_cfg["epochs"] = args.epochs
    if args.batch_size is not None: train_cfg["batch_size"] = args.batch_size
    if args.lr is not None: train_cfg["lr"] = args.lr

    ds = BigSMILESDataset(args.dataset, args.text_column, tk, max_length=model_cfg["max_seq_len"])
    dl = DataLoader(ds, batch_size=train_cfg["batch_size"], shuffle=True,
                    collate_fn=lambda b: collate_batch(b, tk.pad_id()))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lm = CausalTransformerLM(LMConfig(
        vocab_size=tk.vocab_size,
        n_embd=model_cfg["n_embd"],
        n_layer=model_cfg["n_layer"],
        n_head=model_cfg["n_head"],
        max_seq_len=model_cfg["max_seq_len"],
        dropout=model_cfg["dropout"],
        pad_id=tk.pad_id(),
    )).to(device)

    opt = torch.optim.AdamW(lm.parameters(), lr=train_cfg["lr"])
    lm.train()
    global_step = 0
    for epoch in range(train_cfg["epochs"]):
        pbar = tqdm(dl, desc=f"epoch {epoch+1}/{train_cfg['epochs']}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = lm(**batch)
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lm.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            global_step += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # save checkpoint per epoch
        ckpt_path = os.path.join(args.out_dir, f"lm_epoch{epoch+1}.pt")
        torch.save({"model_state": lm.state_dict(), "cfg": lm.cfg.__dict__}, ckpt_path)

    # final
    torch.save({"model_state": lm.state_dict(), "cfg": lm.cfg.__dict__},
               os.path.join(args.out_dir, "lm_final.pt"))

if __name__ == "__main__":
    main()