#!/usr/bin/env bash
set -euo pipefail

TOKENIZER=${1:-artifacts/spm/bigsmiles_spm.model}
MODEL_DIR=${2:-artifacts/lm/gpt2_bigsmiles}
OUT_PATH=${3:-artifacts/samples/samples.txt}
NUM=${4:-100}

python -m bigsmiles_gen.generate.sample_constrained \
  --tokenizer "${TOKENIZER}" \
  --model_dir "${MODEL_DIR}" \
  --num_samples "${NUM}" \
  --out_path "${OUT_PATH}" \
  --top_p 0.95 --top_k 50 --temperature 0.9