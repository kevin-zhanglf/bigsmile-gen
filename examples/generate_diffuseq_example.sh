#!/bin/bash
# Example script for generating sequences with DiffuSeq

# This example assumes you have:
# 1. A trained DiffuSeq model checkpoint
# 2. A trained tokenizer

# Set paths
MODEL_PATH="outputs/diffuseq/diffuseq_final.pt"
TOKENIZER_DIR="outputs/tokenizer"
OUTPUT_FILE="outputs/generated_sequences.txt"

# Generation parameters
NUM_SAMPLES=100
SEQ_LENGTH=128
BATCH_SIZE=16
NUM_STEPS=100  # Fewer steps for faster generation

# Generate sequences
python -m polygen.generate_diffuseq \
    --model-path "$MODEL_PATH" \
    --tokenizer-dir "$TOKENIZER_DIR" \
    --output "$OUTPUT_FILE" \
    --num-samples "$NUM_SAMPLES" \
    --seq-len "$SEQ_LENGTH" \
    --batch-size "$BATCH_SIZE" \
    --num-steps "$NUM_STEPS"

echo "Generation complete! Sequences saved to $OUTPUT_FILE"
