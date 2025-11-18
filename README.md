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