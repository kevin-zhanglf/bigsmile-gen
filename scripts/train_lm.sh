#!/usr/bin/env bash
set -euo pipefail

TOKENIZER=${1:-artifacts/spm/bigsmiles_spm.model}
TRAIN=${2:-data/bigsmiles.csv}
TEXT_COLUMN=${3:-bigsmiles}
OUT_DIR=${4:-artifacts/lm/gpt2_bigsmiles}

python -m bigsmiles_gen.models.train_lm \
  --tokenizer "${TOKENIZER}" \
  --train "${TRAIN}" \
  --text_column "${TEXT_COLUMN}" \
  --output_dir "${OUT_DIR}" \
  --config configs/lm.yaml