import argparse
import os
import pandas as pd
import sentencepiece as spm
from bigsmiles_gen.tokenizer.pretokenize import pretokenize

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV/TSV/TXT with a text column containing BigSMILES")
    ap.add_argument("--text_column", default="bigsmiles")
    ap.add_argument("--model_prefix", required=True)
    ap.add_argument("--vocab_size", type=int, default=8000)
    ap.add_argument("--user_symbols", default="<BOS>,<EOS>,<PAD>,<MASK>,{,},[,],(,),.,-,=,#,>,<,$,*,:,/,+")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.model_prefix), exist_ok=True)

    # Load text
    ext = os.path.splitext(args.input)[1].lower()
    if ext in [".csv", ".tsv"]:
        df = pd.read_csv(args.input) if ext == ".csv" else pd.read_csv(args.input, sep="\t")
        texts = df[args.text_column].astype(str).tolist()
    else:  # raw txt
        with open(args.input, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]

    tmp_in = f"{args.model_prefix}.tmp.txt"
    with open(tmp_in, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(pretokenize(t) + "\n")

    spm.SentencePieceTrainer.Train(
        input=tmp_in,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        character_coverage=1.0,
        model_type="bpe",
        user_defined_symbols=args.user_symbols.split(","),
        input_sentence_size=min(2000000, len(texts)),
        shuffle_input_sentence=True,
    )
    os.remove(tmp_in)
    print(f"Tokenizer saved to {args.model_prefix}.model / .vocab")

if __name__ == "__main__":
    main()