#!/usr/bin/env bash
set -euo pipefail

INPUT=${1:-data/bigsmiles.csv}
TEXT_COLUMN=${2:-bigsmiles}
OUT_PREFIX=${3:-artifacts/spm/bigsmiles_spm}
VOCAB_SIZE=${4:-8000}

python -m bigsmiles_gen.tokenizer.train_tokenizer \
  --input "${INPUT}" \
  --text_column "${TEXT_COLUMN}" \
  --model_prefix "${OUT_PREFIX}" \
  --vocab_size "${VOCAB_SIZE}" \
  --user_symbols "<BOS>,<EOS>,<PAD>,<MASK>,{,},[,],(,),.,-,=,#,>,<,$,*,:,/,+"