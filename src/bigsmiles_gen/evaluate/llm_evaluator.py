"""
LLM-based evaluator for BigSMILES strings using OpenAI and Qwen models.

This module provides functionality to evaluate generated BigSMILES strings using
large language models (OpenAI GPT models and Alibaba Qwen models) to assess:
- Chemical validity and correctness
- Structural coherence
- Polymer representation quality
- Overall quality scores
"""

import os
import argparse
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json


@dataclass
class EvaluationResult:
    """Result of LLM evaluation for a BigSMILES string."""
    bigsmiles: str
    score: float  # 0-1 score
    reasoning: str
    is_valid: bool
    feedback: str


class LLMEvaluator:
    """Base class for LLM-based BigSMILES evaluators."""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        """
        Initialize the LLM evaluator.
        
        Args:
            model_name: Name of the model to use
            api_key: API key for the service (if None, will try to read from environment)
        """
        self.model_name = model_name
        self.api_key = api_key
        
    def create_prompt(self, bigsmiles: str) -> str:
        """
        Create evaluation prompt for the LLM.
        
        Args:
            bigsmiles: BigSMILES string to evaluate
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""You are an expert in polymer chemistry and BigSMILES notation.

Please evaluate the following BigSMILES string for:
1. Chemical validity and correctness
2. Proper use of BigSMILES notation (brackets, bonding descriptors, stochastic objects)
3. Structural coherence and polymer representation quality

BigSMILES string: {bigsmiles}

Provide your evaluation in the following JSON format:
{{
    "score": <float between 0 and 1>,
    "is_valid": <true or false>,
    "reasoning": "<brief explanation of the evaluation>",
    "feedback": "<specific suggestions for improvement if any>"
}}

Respond only with the JSON object, no additional text."""
        return prompt
    
    def parse_response(self, response: str) -> Dict:
        """
        Parse LLM response to extract evaluation results.
        
        Args:
            response: Raw response from LLM
            
        Returns:
            Dictionary with evaluation results
        """
        try:
            # Try to find JSON in the response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                result = json.loads(json_str)
                return result
            else:
                # Fallback if no JSON found
                return {
                    "score": 0.5,
                    "is_valid": True,
                    "reasoning": "Could not parse structured response",
                    "feedback": response[:200]
                }
        except json.JSONDecodeError:
            return {
                "score": 0.5,
                "is_valid": True,
                "reasoning": "Failed to parse JSON response",
                "feedback": response[:200]
            }
    
    def evaluate(self, bigsmiles: str) -> EvaluationResult:
        """
        Evaluate a single BigSMILES string.
        
        Args:
            bigsmiles: BigSMILES string to evaluate
            
        Returns:
            EvaluationResult object
        """
        raise NotImplementedError("Subclasses must implement evaluate()")
    
    def evaluate_batch(self, bigsmiles_list: List[str]) -> List[EvaluationResult]:
        """
        Evaluate a batch of BigSMILES strings.
        
        Args:
            bigsmiles_list: List of BigSMILES strings
            
        Returns:
            List of EvaluationResult objects
        """
        results = []
        for bs in bigsmiles_list:
            result = self.evaluate(bs)
            results.append(result)
        return results


class OpenAIEvaluator(LLMEvaluator):
    """Evaluator using OpenAI models (GPT-3.5, GPT-4, etc.)."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", api_key: Optional[str] = None):
        """
        Initialize OpenAI evaluator.
        
        Args:
            model_name: OpenAI model name (e.g., "gpt-3.5-turbo", "gpt-4")
            api_key: OpenAI API key (if None, reads from OPENAI_API_KEY env var)
        """
        super().__init__(model_name, api_key)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided and OPENAI_API_KEY environment variable not set")
        
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("openai package not installed. Install with: pip install openai")
    
    def evaluate(self, bigsmiles: str) -> EvaluationResult:
        """
        Evaluate a BigSMILES string using OpenAI model.
        
        Args:
            bigsmiles: BigSMILES string to evaluate
            
        Returns:
            EvaluationResult object
        """
        try:
            prompt = self.create_prompt(bigsmiles)
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert in polymer chemistry and BigSMILES notation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            content = response.choices[0].message.content
            result = self.parse_response(content)
            
            return EvaluationResult(
                bigsmiles=bigsmiles,
                score=float(result.get("score", 0.5)),
                reasoning=result.get("reasoning", ""),
                is_valid=result.get("is_valid", True),
                feedback=result.get("feedback", "")
            )
        except Exception as e:
            return EvaluationResult(
                bigsmiles=bigsmiles,
                score=0.0,
                reasoning=f"Error during evaluation: {str(e)}",
                is_valid=False,
                feedback=""
            )


class QwenEvaluator(LLMEvaluator):
    """Evaluator using Alibaba Qwen models."""
    
    def __init__(self, model_name: str = "qwen-turbo", api_key: Optional[str] = None):
        """
        Initialize Qwen evaluator.
        
        Args:
            model_name: Qwen model name (e.g., "qwen-turbo", "qwen-plus", "qwen-max")
            api_key: DashScope API key (if None, reads from DASHSCOPE_API_KEY env var)
        """
        super().__init__(model_name, api_key)
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("DashScope API key not provided and DASHSCOPE_API_KEY environment variable not set")
        
        try:
            import dashscope
            dashscope.api_key = self.api_key
            self.dashscope = dashscope
        except ImportError:
            raise ImportError("dashscope package not installed. Install with: pip install dashscope")
    
    def evaluate(self, bigsmiles: str) -> EvaluationResult:
        """
        Evaluate a BigSMILES string using Qwen model.
        
        Args:
            bigsmiles: BigSMILES string to evaluate
            
        Returns:
            EvaluationResult object
        """
        try:
            from dashscope import Generation
            
            prompt = self.create_prompt(bigsmiles)
            
            response = Generation.call(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert in polymer chemistry and BigSMILES notation."},
                    {"role": "user", "content": prompt}
                ],
                result_format="message",
                temperature=0.3,
                max_tokens=500
            )
            
            if response.status_code == 200:
                content = response.output.choices[0].message.content
                result = self.parse_response(content)
                
                return EvaluationResult(
                    bigsmiles=bigsmiles,
                    score=float(result.get("score", 0.5)),
                    reasoning=result.get("reasoning", ""),
                    is_valid=result.get("is_valid", True),
                    feedback=result.get("feedback", "")
                )
            else:
                return EvaluationResult(
                    bigsmiles=bigsmiles,
                    score=0.0,
                    reasoning=f"API error: {response.code} - {response.message}",
                    is_valid=False,
                    feedback=""
                )
        except Exception as e:
            return EvaluationResult(
                bigsmiles=bigsmiles,
                score=0.0,
                reasoning=f"Error during evaluation: {str(e)}",
                is_valid=False,
                feedback=""
            )


def create_evaluator(provider: str, model_name: Optional[str] = None, api_key: Optional[str] = None) -> LLMEvaluator:
    """
    Factory function to create an evaluator instance.
    
    Args:
        provider: "openai" or "qwen"
        model_name: Model name (optional, uses defaults if not provided)
        api_key: API key (optional, reads from environment if not provided)
        
    Returns:
        LLMEvaluator instance
    """
    if provider.lower() == "openai":
        model = model_name or "gpt-3.5-turbo"
        return OpenAIEvaluator(model_name=model, api_key=api_key)
    elif provider.lower() == "qwen":
        model = model_name or "qwen-turbo"
        return QwenEvaluator(model_name=model, api_key=api_key)
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'qwen'")


def main():
    """Command-line interface for LLM evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate BigSMILES strings using LLM (OpenAI or Qwen)"
    )
    parser.add_argument(
        "--input", 
        required=True, 
        help="Input file with BigSMILES strings (one per line)"
    )
    parser.add_argument(
        "--output", 
        required=True, 
        help="Output JSON file with evaluation results"
    )
    parser.add_argument(
        "--provider", 
        choices=["openai", "qwen"], 
        required=True,
        help="LLM provider to use"
    )
    parser.add_argument(
        "--model", 
        help="Model name (optional, uses defaults)"
    )
    parser.add_argument(
        "--api_key", 
        help="API key (optional, reads from environment)"
    )
    parser.add_argument(
        "--min_score",
        type=float,
        default=0.0,
        help="Minimum score threshold to include in output (0-1)"
    )
    
    args = parser.parse_args()
    
    # Create evaluator
    print(f"Initializing {args.provider} evaluator...")
    evaluator = create_evaluator(args.provider, args.model, args.api_key)
    
    # Read input
    print(f"Reading BigSMILES from {args.input}...")
    with open(args.input, "r", encoding="utf-8") as f:
        bigsmiles_list = [line.strip() for line in f if line.strip()]
    
    print(f"Evaluating {len(bigsmiles_list)} BigSMILES strings...")
    
    # Evaluate
    results = []
    for i, bs in enumerate(bigsmiles_list, 1):
        print(f"  [{i}/{len(bigsmiles_list)}] Evaluating: {bs[:50]}...")
        result = evaluator.evaluate(bs)
        
        if result.score >= args.min_score:
            results.append({
                "bigsmiles": result.bigsmiles,
                "score": result.score,
                "is_valid": result.is_valid,
                "reasoning": result.reasoning,
                "feedback": result.feedback
            })
    
    # Write output
    print(f"Writing results to {args.output}...")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"\nEvaluation complete!")
    print(f"Total evaluated: {len(bigsmiles_list)}")
    print(f"Passed threshold (>= {args.min_score}): {len(results)}")
    if results:
        avg_score = sum(r["score"] for r in results) / len(results)
        print(f"Average score: {avg_score:.3f}")
        valid_count = sum(1 for r in results if r["is_valid"])
        print(f"Valid according to LLM: {valid_count}/{len(results)}")


if __name__ == "__main__":
    main()
