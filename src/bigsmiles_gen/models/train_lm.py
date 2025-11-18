import argparse, os, yaml, random, numpy as np, torch
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          DataCollatorForLanguageModeling, Trainer, TrainingArguments)
from bigsmiles_gen.tokenizer.pretokenize import pretokenize
from bigsmiles_gen.data.dataset import BigSMILESTextDataset

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--text_column", default="bigsmiles")
    ap.add_argument("--prefix_column", default=None)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--config", default="configs/lm.yaml")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    os.makedirs(args.output_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=True, use_fast=False
    )
    # Special tokens (ensure present)
    special_add = {"bos_token": "<BOS>", "eos_token": "<EOS>", "pad_token": "<PAD>"}
    for k, v in special_add.items():
        if getattr(tok, k, None) is None or tok.convert_tokens_to_ids(v) == tok.unk_token_id:
            tok.add_special_tokens({"additional_special_tokens": [v]})
    if tok.pad_token is None:
        tok.pad_token = "<PAD>"

    ds_py = BigSMILESTextDataset(args.train, text_column=args.text_column, prefix_column=args.prefix_column)
    texts = [pretokenize(x["text"]) for x in ds_py]
    ds = Dataset.from_dict({"text": texts})

    def tok_fn(batch):
        return tok(batch["text"], truncation=True, max_length=cfg.get("block_size", 512))

    tokenized = ds.map(tok_fn, batched=True, remove_columns=["text"])

    model = AutoModelForCausalLM.from_pretrained(cfg.get("model_name", "gpt2"))
    model.resize_token_embeddings(len(tok))

    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=cfg.get("batch_size", 8),
        num_train_epochs=cfg.get("epochs", 3),
        learning_rate=cfg.get("learning_rate", 5e-5),
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        weight_decay=cfg.get("weight_decay", 0.01),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
        save_steps=cfg.get("save_steps", 500),
        eval_steps=cfg.get("eval_steps", 500),
        logging_steps=cfg.get("logging_steps", 50),
        save_total_limit=cfg.get("save_total_limit", 2),
        bf16=torch.cuda.is_available(),
        report_to=[],
    )

    trainer = Trainer(
        model=model, args=targs, train_dataset=tokenized, data_collator=collator
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()