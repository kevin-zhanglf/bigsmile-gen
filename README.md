# bigsmile-gen

BigSMILES generative modeling toolkit with LLM-based evaluation.

## Features

- Train custom language models for BigSMILES generation
- Grammar-constrained sampling for valid polymer structures
- LLM-based evaluation using OpenAI (GPT) and Qwen models
- Property-based screening and filtering
- Validation and quality assessment

## LLM Evaluator (NEW)

Evaluate generated BigSMILES strings using large language models:

- **OpenAI Integration**: GPT-3.5, GPT-4, etc.
- **Qwen Integration**: Alibaba Cloud Qwen models
- **Quality Assessment**: Chemical validity, structural coherence, notation correctness

See [LLM Evaluator Documentation](docs/LLM_EVALUATOR.md) for details.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Evaluate BigSMILES with LLM
export OPENAI_API_KEY='your-key'
python -m bigsmiles_gen.evaluate.llm_evaluator \
    --input samples.txt \
    --output results.json \
    --provider openai \
    --min_score 0.6
```

For more examples, see the [examples/](examples/) directory.
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
