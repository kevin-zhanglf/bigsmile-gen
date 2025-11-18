# LLM Evaluator for BigSMILES Generation

This module provides LLM-based evaluation of generated BigSMILES strings using OpenAI GPT models and Alibaba Qwen models.

## Features

- **OpenAI Integration**: Use GPT-3.5, GPT-4, or other OpenAI models to evaluate BigSMILES strings
- **Qwen Integration**: Use Alibaba Cloud Qwen models (qwen-turbo, qwen-plus, qwen-max)
- **Quality Assessment**: Evaluate chemical validity, structural coherence, and notation correctness
- **Batch Processing**: Evaluate multiple BigSMILES strings efficiently
- **Configurable**: Flexible configuration via YAML or command-line arguments

## Installation

Install the required dependencies:

```bash
pip install openai>=1.0.0      # For OpenAI models
pip install dashscope>=1.0.0   # For Qwen models
```

Or install from requirements.txt:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Command-Line Interface

#### Using OpenAI (GPT models)

Set your API key:
```bash
export OPENAI_API_KEY='your-openai-api-key'
```

Evaluate BigSMILES strings:
```bash
python -m bigsmiles_gen.evaluate.llm_evaluator \
    --input samples.txt \
    --output results.json \
    --provider openai \
    --model gpt-3.5-turbo \
    --min_score 0.6
```

#### Using Qwen (Alibaba Cloud)

Set your API key:
```bash
export DASHSCOPE_API_KEY='your-dashscope-api-key'
```

Evaluate BigSMILES strings:
```bash
python -m bigsmiles_gen.evaluate.llm_evaluator \
    --input samples.txt \
    --output results.json \
    --provider qwen \
    --model qwen-turbo \
    --min_score 0.6
```

#### Using the Shell Script

```bash
# Using OpenAI
./scripts/evaluate_with_llm.sh samples.txt openai gpt-3.5-turbo

# Using Qwen
./scripts/evaluate_with_llm.sh samples.txt qwen qwen-turbo
```

### 2. Python API

```python
from bigsmiles_gen.evaluate.llm_evaluator import create_evaluator

# Create an evaluator
evaluator = create_evaluator(
    provider="openai",
    model_name="gpt-3.5-turbo",
    api_key="your-api-key"  # Optional, reads from environment
)

# Evaluate a single BigSMILES string
result = evaluator.evaluate("{[][<]CC[>][<]CC(C)[>][]}")

print(f"Score: {result.score}")
print(f"Valid: {result.is_valid}")
print(f"Reasoning: {result.reasoning}")
print(f"Feedback: {result.feedback}")

# Evaluate multiple strings
bigsmiles_list = [
    "{[][<]CC[>][<]CC(C)[>][]}",
    "{[][$]CC[$][]}"
]
results = evaluator.evaluate_batch(bigsmiles_list)

for r in results:
    print(f"{r.bigsmiles}: score={r.score:.2f}, valid={r.is_valid}")
```

### 3. Using Different Models

#### OpenAI Models
- `gpt-3.5-turbo`: Fast and cost-effective
- `gpt-4`: More accurate but slower
- `gpt-4-turbo`: Balance of speed and accuracy

#### Qwen Models
- `qwen-turbo`: Fast inference
- `qwen-plus`: Better performance
- `qwen-max`: Highest quality

## Configuration

Edit `configs/llm_evaluator.yaml` to customize evaluation settings:

```yaml
# Provider: "openai" or "qwen"
provider: openai

# Model settings
openai:
  model: gpt-3.5-turbo
  api_key_env: OPENAI_API_KEY

qwen:
  model: qwen-turbo
  api_key_env: DASHSCOPE_API_KEY

# Evaluation settings
evaluation:
  min_score: 0.6
  temperature: 0.3
  max_tokens: 500
```

## Output Format

The evaluation produces a JSON file with the following structure:

```json
[
  {
    "bigsmiles": "{[][<]CC[>][<]CC(C)[>][]}",
    "score": 0.85,
    "is_valid": true,
    "reasoning": "Valid BigSMILES notation with proper stochastic objects and bonding descriptors",
    "feedback": "Structure looks chemically sound"
  },
  {
    "bigsmiles": "{[][$]CC[$][]}",
    "score": 0.72,
    "is_valid": true,
    "reasoning": "Simple but valid structure",
    "feedback": "Could be more specific with bonding descriptors"
  }
]
```

## Integration with Pipeline

The LLM evaluator can be integrated into the existing pipeline as an additional validation step:

```bash
# 1. Generate samples
python -m bigsmiles_gen.generate.sample_constrained \
    --tokenizer artifacts/spm/bigsmiles_spm.model \
    --model_dir artifacts/lm/gpt2_bigsmiles \
    --num_samples 100 \
    --out_path artifacts/samples/samples.txt

# 2. Basic validation
python -m bigsmiles_gen.validate.validator \
    --input artifacts/samples/samples.txt

# 3. LLM evaluation (NEW)
python -m bigsmiles_gen.evaluate.llm_evaluator \
    --input artifacts/samples/samples.txt \
    --output artifacts/samples/evaluated.json \
    --provider openai \
    --min_score 0.7

# 4. Property screening (existing)
python -m bigsmiles_gen.score.property_screener \
    --input artifacts/samples/samples.txt \
    --props data/properties.csv \
    --rules configs/screening.yaml \
    --out artifacts/samples/screened.txt
```

## API Keys

### OpenAI
1. Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)
2. Set the environment variable: `export OPENAI_API_KEY='your-key'`

### Qwen (DashScope)
1. Get your API key from [Alibaba Cloud DashScope](https://dashscope.console.aliyun.com/)
2. Set the environment variable: `export DASHSCOPE_API_KEY='your-key'`

⚠️ **Security Note**: Never commit API keys to version control. Always use environment variables or secure key management systems.

## Cost Considerations

- **OpenAI**: Charges per token (input + output). GPT-3.5-turbo is most cost-effective.
- **Qwen**: Charges based on model and tokens. Check [DashScope pricing](https://help.aliyun.com/zh/dashscope/developer-reference/tongyi-qianwen-metering-and-billing).

For large-scale evaluation, consider:
- Using batch API endpoints if available
- Caching results for repeated evaluations
- Using less expensive models for initial filtering

## Troubleshooting

### "API key not provided" error
- Ensure you've set the appropriate environment variable (`OPENAI_API_KEY` or `DASHSCOPE_API_KEY`)
- Check that the variable is exported in your current shell session

### Rate limiting errors
- Add delays between API calls
- Use batch processing with appropriate rate limits
- Consider using higher tier API plans

### JSON parsing errors
- The module includes fallback parsing for malformed responses
- Check model temperature settings (lower values produce more consistent outputs)

## Contributing

When adding new LLM providers:
1. Extend the `LLMEvaluator` base class
2. Implement the `evaluate()` method
3. Add provider configuration to `configs/llm_evaluator.yaml`
4. Update this documentation

## References

- [BigSMILES Documentation](https://github.com/IntelLabs/bigSMILES)
- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)
- [DashScope Documentation](https://help.aliyun.com/zh/dashscope/)
