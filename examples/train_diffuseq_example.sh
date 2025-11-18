#!/bin/bash
# Example script for training DiffuSeq model

# This example assumes you have:
# 1. A dataset CSV file (e.g., data/train.csv)
# 2. A trained tokenizer in outputs/tokenizer/

# Set paths
DATASET="data/train.csv"
TEXT_COLUMN="bigsmiles"
TOKENIZER_DIR="outputs/tokenizer"
OUTPUT_DIR="outputs/diffuseq"
CONFIG="configs/diffuseq.yaml"

# Training parameters
EPOCHS=10
BATCH_SIZE=16
LEARNING_RATE=1e-4
NUM_DIFFUSION_STEPS=2000

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Train DiffuSeq model
python -m polygen.train_diffuseq \
    --dataset "$DATASET" \
    --text-column "$TEXT_COLUMN" \
    --tokenizer-dir "$TOKENIZER_DIR" \
    --out-dir "$OUTPUT_DIR" \
    --config "$CONFIG" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LEARNING_RATE" \
    --num-diffusion-steps "$NUM_DIFFUSION_STEPS"

echo "Training complete! Model saved to $OUTPUT_DIR"
