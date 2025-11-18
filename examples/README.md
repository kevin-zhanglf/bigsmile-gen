# Examples

This directory contains examples demonstrating the use of the LLM evaluator for BigSMILES strings.

## Files

- **test_bigsmiles.txt**: Sample BigSMILES strings for testing
- **evaluate_example.py**: Python script demonstrating how to use the evaluator API

## Running the Example

### Prerequisites

1. Install dependencies:
   ```bash
   pip install -r ../requirements.txt
   ```

2. Set up API key (choose one):
   - For OpenAI: `export OPENAI_API_KEY='your-key'`
   - For Qwen: `export DASHSCOPE_API_KEY='your-key'`

### Using the Python Script

```bash
python evaluate_example.py
```

### Using the Command-Line Tool

```bash
# With OpenAI
PYTHONPATH=../src python -m bigsmiles_gen.evaluate.llm_evaluator \
    --input test_bigsmiles.txt \
    --output results.json \
    --provider openai \
    --min_score 0.6

# With Qwen
PYTHONPATH=../src python -m bigsmiles_gen.evaluate.llm_evaluator \
    --input test_bigsmiles.txt \
    --output results.json \
    --provider qwen \
    --min_score 0.6
```

### Using the Shell Script

```bash
../scripts/evaluate_with_llm.sh test_bigsmiles.txt openai gpt-3.5-turbo
```

## Output

The evaluation will produce a JSON file with scores, validity assessments, reasoning, and feedback for each BigSMILES string.

Example output:
```json
[
  {
    "bigsmiles": "{[][<]CC[>][<]CC(C)[>][]}",
    "score": 0.85,
    "is_valid": true,
    "reasoning": "Valid BigSMILES notation with proper stochastic objects",
    "feedback": "Structure looks chemically sound"
  }
]
```

For more details, see the [LLM Evaluator Documentation](../docs/LLM_EVALUATOR.md).
