# bigsmile-gen

Minimal scaffold for BigSMILES generative modeling with support for multiple generation approaches.

## Features

### Autoregressive Language Models
- Traditional causal transformer models (GPT-style)
- Left-to-right sequence generation
- Located in `src/bigsmiles_gen/` and `src/polygen/`

### DiffuSeq: Discrete Diffusion Models
- Non-autoregressive sequence generation
- Token-level diffusion with parallel decoding
- Iterative denoising for high-quality generation
- See [DiffuSeq Documentation](docs/DIFFUSEQ.md) for details

## Quick Start

### DiffuSeq Training

```bash
# Train DiffuSeq model
python -m polygen.train_diffuseq \
    --dataset data/train.csv \
    --text-column bigsmiles \
    --tokenizer-dir outputs/tokenizer \
    --out-dir outputs/diffuseq \
    --config configs/diffuseq.yaml
```

### DiffuSeq Generation

```bash
# Generate sequences
python -m polygen.generate_diffuseq \
    --model-path outputs/diffuseq/diffuseq_final.pt \
    --tokenizer-dir outputs/tokenizer \
    --output generated_sequences.txt \
    --num-samples 100
```

## Documentation

- [DiffuSeq Implementation Guide](docs/DIFFUSEQ.md)
- Example scripts in `examples/`

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Repository Structure

```
bigsmile-gen/
├── src/
│   ├── bigsmiles_gen/      # BigSMILES-specific tools
│   └── polygen/             # Polymer generation models
│       ├── diffuseq_model.py       # DiffuSeq implementation
│       ├── diffuseq_dataset.py     # DiffuSeq dataset
│       ├── train_diffuseq.py       # Training script
│       └── generate_diffuseq.py    # Generation script
├── configs/
│   ├── lm.yaml             # Autoregressive LM config
│   └── diffuseq.yaml       # DiffuSeq config
├── docs/
│   └── DIFFUSEQ.md         # DiffuSeq documentation
└── examples/               # Example scripts
```