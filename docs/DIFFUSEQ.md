# DiffuSeq Implementation

This document describes the DiffuSeq (Discrete Diffusion for Sequence Generation) implementation in this repository.

## Overview

DiffuSeq is a discrete diffusion model for non-autoregressive sequence generation. Unlike traditional autoregressive language models that generate tokens one by one, DiffuSeq generates the entire sequence in parallel through an iterative denoising process.

## Key Concepts

### 1. Discrete Diffusion Process

**Forward Process (Adding Noise):**
- Start with clean token sequence `x_0`
- Gradually mask tokens according to a diffusion schedule
- At timestep `t`, tokens are masked with probability determined by the schedule
- At final timestep `T`, all tokens are masked

**Reverse Process (Denoising):**
- Start with fully masked sequence `x_T`
- Iteratively predict and fill in masked tokens
- Use a transformer model to predict original tokens from noisy input
- At timestep 0, obtain the final generated sequence `x_0`

### 2. Diffusion Schedule

Two schedule types are supported:

- **Linear Schedule**: Uniform rate of noise addition
- **Cosine Schedule**: More gradual at start and end (recommended)

The schedule determines the masking probability at each timestep:
- `t=0`: No masking (clean data)
- `t=T`: Full masking (all tokens masked)

### 3. Denoising Transformer

The model architecture:
- **Bidirectional Transformer**: Unlike causal LM, can attend to all positions
- **Time Embeddings**: Sinusoidal embeddings encode the current timestep
- **Token + Position + Time**: Combined embeddings for rich representation
- **Output**: Predicts original token at each position

## Architecture

```
Input: [noisy_tokens, timestep]
  ↓
Token Embedding + Position Embedding + Time Embedding
  ↓
Bidirectional Transformer Layers (no causal mask)
  ↓
Layer Norm
  ↓
Linear Head → Logits over vocabulary
```

## Files

- `src/polygen/diffuseq_model.py`: Core model implementation
  - `DiffuSeqConfig`: Model configuration
  - `DiffusionSchedule`: Noise schedule
  - `DiffuSeqModel`: Main denoising model
  
- `src/polygen/diffuseq_dataset.py`: Dataset and collation
  - `DiffuSeqDataset`: Dataset wrapper
  - `collate_diffuseq_batch`: Batch collation with padding

- `src/polygen/train_diffuseq.py`: Training script
- `src/polygen/generate_diffuseq.py`: Generation script
- `configs/diffuseq.yaml`: Default configuration

## Usage

### 1. Prepare Data

Ensure you have:
- CSV file with text sequences
- Trained tokenizer (see tokenizer training in polygen)

### 2. Train DiffuSeq Model

```bash
python -m polygen.train_diffuseq \
    --dataset data/train.csv \
    --text-column bigsmiles \
    --tokenizer-dir outputs/tokenizer \
    --out-dir outputs/diffuseq \
    --config configs/diffuseq.yaml \
    --epochs 10 \
    --batch-size 16
```

### 3. Generate Sequences

```bash
python -m polygen.generate_diffuseq \
    --model-path outputs/diffuseq/diffuseq_final.pt \
    --tokenizer-dir outputs/tokenizer \
    --output outputs/generated_sequences.txt \
    --num-samples 100 \
    --seq-len 128 \
    --num-steps 100  # Fewer steps = faster but lower quality
```

## Training Details

### Loss Function

The training loss is computed only on masked positions:

```python
loss = CrossEntropy(predicted_tokens[masked_positions], 
                   original_tokens[masked_positions])
```

### Training Procedure

1. Sample random timestep `t` for each sequence in batch
2. Apply forward diffusion to create noisy input (mask tokens)
3. Predict original tokens using denoising model
4. Compute loss on masked positions
5. Backpropagate and update weights

### Hyperparameters

Key hyperparameters (in `configs/diffuseq.yaml`):
- `num_steps`: Number of diffusion timesteps (default: 2000)
- `schedule_type`: Noise schedule type ('cosine' recommended)
- `n_layer`: Number of transformer layers (default: 6)
- `n_embd`: Embedding dimension (default: 384)
- `lr`: Learning rate (default: 1e-4)

## Generation Details

### Iterative Denoising

1. Start with all tokens masked: `x_T = [MASK, MASK, ..., MASK]`
2. For timestep `t` from `T-1` down to 0:
   - Predict original tokens at masked positions
   - Sample from predicted distribution
   - Optionally remask some tokens for next iteration
3. Return final denoised sequence `x_0`

### Generation Speed

- **Full denoising** (num_steps=2000): Highest quality, slower
- **Accelerated** (num_steps=100-500): Faster, slightly lower quality
- Generation is parallelized across sequence positions

## Advantages over Autoregressive Models

1. **Parallel Generation**: All positions decoded simultaneously
2. **Flexible Conditioning**: Can condition on partial sequences
3. **Iterative Refinement**: Can improve samples with more steps
4. **No Left-to-Right Bias**: Bidirectional context

## Comparison with MaskGIT

DiffuSeq shares similarities with MaskGIT:
- Both use token masking/unmasking
- Both are non-autoregressive
- MaskGIT is typically used for images, DiffuSeq for text

Key difference:
- MaskGIT uses confidence-based unmasking
- DiffuSeq uses diffusion schedule for masking probability

## References

This implementation is based on:
- **DiffuSeq**: "DiffuSeq: Sequence to Sequence Text Generation with Diffusion Models"
- **MaskGIT**: "MaskGIT: Masked Generative Image Transformer"
- Discrete diffusion models for text generation

## Future Improvements

Possible enhancements:
1. Classifier-free guidance for controlled generation
2. Conditional generation with prefix/context
3. Faster sampling strategies (DDIM-style)
4. Hybrid autoregressive-diffusion models
5. Adaptive number of denoising steps per position
