#!/bin/bash
# Example script to evaluate BigSMILES strings using LLM

set -e

# Check if input file is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <input_file> [provider] [model]"
    echo "  input_file: Text file with BigSMILES strings (one per line)"
    echo "  provider: 'openai' or 'qwen' (default: openai)"
    echo "  model: Model name (optional)"
    echo ""
    echo "Examples:"
    echo "  $0 samples.txt openai gpt-3.5-turbo"
    echo "  $0 samples.txt qwen qwen-turbo"
    exit 1
fi

INPUT_FILE="$1"
PROVIDER="${2:-openai}"
MODEL="${3:-}"
OUTPUT_FILE="${INPUT_FILE%.txt}_evaluated.json"

echo "Evaluating BigSMILES from: $INPUT_FILE"
echo "Provider: $PROVIDER"
if [ -n "$MODEL" ]; then
    echo "Model: $MODEL"
fi
echo "Output: $OUTPUT_FILE"
echo ""

# Check API key based on provider
if [ "$PROVIDER" = "openai" ]; then
    if [ -z "$OPENAI_API_KEY" ]; then
        echo "Error: OPENAI_API_KEY environment variable not set"
        echo "Please set it with: export OPENAI_API_KEY='your-api-key'"
        exit 1
    fi
elif [ "$PROVIDER" = "qwen" ]; then
    if [ -z "$DASHSCOPE_API_KEY" ]; then
        echo "Error: DASHSCOPE_API_KEY environment variable not set"
        echo "Please set it with: export DASHSCOPE_API_KEY='your-api-key'"
        exit 1
    fi
else
    echo "Error: Unknown provider '$PROVIDER'. Use 'openai' or 'qwen'"
    exit 1
fi

# Run evaluation
if [ -n "$MODEL" ]; then
    python -m bigsmiles_gen.evaluate.llm_evaluator \
        --input "$INPUT_FILE" \
        --output "$OUTPUT_FILE" \
        --provider "$PROVIDER" \
        --model "$MODEL" \
        --min_score 0.6
else
    python -m bigsmiles_gen.evaluate.llm_evaluator \
        --input "$INPUT_FILE" \
        --output "$OUTPUT_FILE" \
        --provider "$PROVIDER" \
        --min_score 0.6
fi

echo ""
echo "Evaluation complete! Results saved to: $OUTPUT_FILE"
