#!/usr/bin/env python3
"""
Example script demonstrating how to use the LLM evaluator for BigSMILES strings.

This script shows:
1. How to create an evaluator instance
2. How to evaluate a single BigSMILES string
3. How to evaluate multiple strings in batch
4. How to handle the evaluation results

Note: You need to set the appropriate API key environment variable:
- For OpenAI: export OPENAI_API_KEY='your-key'
- For Qwen: export DASHSCOPE_API_KEY='your-key'
"""

import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from bigsmiles_gen.evaluate.llm_evaluator import create_evaluator


def main():
    print("=" * 70)
    print("LLM Evaluator Example for BigSMILES")
    print("=" * 70)
    print()
    
    # Check which provider to use based on available API keys
    if os.environ.get("OPENAI_API_KEY"):
        provider = "openai"
        model = "gpt-3.5-turbo"
        print(f"Using OpenAI ({model})")
    elif os.environ.get("DASHSCOPE_API_KEY"):
        provider = "qwen"
        model = "qwen-turbo"
        print(f"Using Qwen ({model})")
    else:
        print("ERROR: No API key found!")
        print("Please set either:")
        print("  - OPENAI_API_KEY for OpenAI models")
        print("  - DASHSCOPE_API_KEY for Qwen models")
        return 1
    
    print()
    
    # Example BigSMILES strings
    bigsmiles_examples = [
        "{[][<]CC[>][<]CC(C)[>][]}",  # Valid copolymer
        "{[][$]CC[$][]}",              # Simple polymer
        "{[]CC(C)(CC[>])CC(C)(C[<])[]}",  # Branched polymer
    ]
    
    try:
        # Create evaluator
        print(f"Creating {provider} evaluator...")
        evaluator = create_evaluator(provider=provider, model_name=model)
        print("✓ Evaluator created successfully")
        print()
        
        # Evaluate single string
        print("-" * 70)
        print("Example 1: Evaluating a single BigSMILES string")
        print("-" * 70)
        test_bigsmiles = bigsmiles_examples[0]
        print(f"BigSMILES: {test_bigsmiles}")
        print()
        
        result = evaluator.evaluate(test_bigsmiles)
        
        print(f"Score:     {result.score:.2f}")
        print(f"Valid:     {result.is_valid}")
        print(f"Reasoning: {result.reasoning}")
        if result.feedback:
            print(f"Feedback:  {result.feedback}")
        print()
        
        # Evaluate batch
        print("-" * 70)
        print("Example 2: Evaluating multiple BigSMILES strings")
        print("-" * 70)
        print(f"Evaluating {len(bigsmiles_examples)} BigSMILES strings...")
        print()
        
        results = evaluator.evaluate_batch(bigsmiles_examples)
        
        for i, result in enumerate(results, 1):
            print(f"Result {i}:")
            print(f"  BigSMILES: {result.bigsmiles}")
            print(f"  Score:     {result.score:.2f}")
            print(f"  Valid:     {result.is_valid}")
            print(f"  Reasoning: {result.reasoning[:100]}...")
            print()
        
        # Summary statistics
        print("-" * 70)
        print("Summary")
        print("-" * 70)
        avg_score = sum(r.score for r in results) / len(results)
        valid_count = sum(1 for r in results if r.is_valid)
        print(f"Total evaluated:  {len(results)}")
        print(f"Average score:    {avg_score:.2f}")
        print(f"Valid structures: {valid_count}/{len(results)}")
        print()
        
        print("✓ Evaluation completed successfully!")
        return 0
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
