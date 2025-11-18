import argparse, os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor
from bigsmiles_gen.grammar.grammar_mask import GrammarMasker

class GrammarProcessor(LogitsProcessor):
    def __init__(self, masker: GrammarMasker):
        self.masker = masker
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        return self.masker.mask(input_ids, scores)

def load_token_id_map(tokenizer) -> dict:
    symbols = ["{","}","[","]","(",")"]
    m = {}
    for s in symbols:
        tid = tokenizer.convert_tokens_to_ids(s)
        if tid != tokenizer.unk_token_id:
            m[s] = tid
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--num_samples", type=int, default=50)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--prefix", default="")
    ap.add_argument("--out_path", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir).eval()
    if torch.cuda.is_available():
        model = model.to("cuda")

    token_id_map = load_token_id_map(tok)
    masker = GrammarMasker(token_id_map, tok.eos_token_id)
    logits_processor = torch.nn.ModuleList([GrammarProcessor(masker)])

    prompts = [args.prefix.strip()] * args.num_samples
    inputs = tok(prompts, return_tensors="pt", padding=True)
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            do_sample=True,
            top_p=args.top_p,
            top_k=args.top_k,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            logits_processor=logits_processor,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )

    decoded = tok.batch_decode(out, skip_special_tokens=True)
    decoded = [line.strip() for line in decoded]

    from bigsmiles_gen.validate.validator import validate_bigsmiles
    valid = []
    for s in decoded:
        ok, reason = validate_bigsmiles(s)
        if ok:
            valid.append(s)

    with open(args.out_path, "w", encoding="utf-8") as f:
        for s in valid:
            f.write(s + "\n")
    print(f"Wrote {len(valid)} valid candidates to {args.out_path} (from {len(decoded)} decoded)")

if __name__ == "__main__":
    main()