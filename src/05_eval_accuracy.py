#!/usr/bin/env python3
"""
Evaluates task accuracy with newly generated chains from the fine-tuned model.
Extracts final answer letters/numbers and compares them to ground truth.
"""

import torch
import json
import os
import re
import argparse
from tqdm import tqdm
from vllm import LLM, SamplingParams
from datasets import load_dataset
import numpy as np

def build_prompt(question, options=None):
    """Build prompt template for task evaluation."""
    if options:
        prompt = f"""Question: {question}

Options: {options}

Think step-by-step to solve this problem."""
    else:
        prompt = f"""Question: {question}

Think step-by-step to solve this problem."""
    return prompt

def extract_answer(cot_text):
    """Extract the final answer letter or number from the chain of thought."""
    # Look for "Answer: X" or "(X)" pattern at the end
    answer_pattern = r'Answer:\s*([A-E]|\d+)'
    alt_pattern = r'\(([A-E]|\d+)\)'
    
    answer_match = re.search(answer_pattern, cot_text)
    if answer_match:
        return answer_match.group(1)
    
    alt_match = re.search(alt_pattern, cot_text)
    if alt_match:
        return alt_match.group(1)
    
    # If no clear answer found, return empty
    return ""

def main():
    parser = argparse.ArgumentParser(description="Evaluate model accuracy")
    parser.add_argument("--model_list", type=str, required=True, help="Path to models.json config file")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing fine-tuned models")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["aqua_rat", "arc_easy"],
        help="Dataset(s) to evaluate on"
    )
    parser.add_argument("--samples", type=int, default=500, help="Number of samples to process")
    parser.add_argument("--output_dir", type=str, default="logs/accuracy", help="Directory to save results")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    torch.manual_seed(0)
    
    # Load model configurations
    import json as j
    import glob
    model_cfgs = j.load(open(args.model_list))
    
    # Process each dataset and model
    for dataset in args.datasets:
        print(f"Processing dataset: {dataset}")
        
        # Load dataset based on name
        try:
            if dataset == "aqua_rat":
                ds = load_dataset("aqua_rat", split="test").shuffle(seed=42)[:args.samples]
            elif dataset == "arc_easy" or dataset == "arc":
                ds = load_dataset("ai2_arc", "ARC-Easy", split="test").shuffle(seed=42)[:args.samples]
            elif dataset == "gsm8k":
                ds = load_dataset("gsm8k", "main", split="test").shuffle(seed=42)[:args.samples]
            elif dataset == "mmlu" or dataset == "mmlu_math":
                # Use a subset of MMLU focused on math
                ds = load_dataset("cais/mmlu", "mathematics", split="test").shuffle(seed=42)[:args.samples]
            elif dataset == "cqna":
                # CommonsenseQA dataset
                ds = load_dataset("commonsense_qa", split="validation").shuffle(seed=42)[:args.samples]
            else:
                print(f"Dataset {dataset} not supported. Skipping.")
                continue
            print(f"Loaded {len(ds)} examples from {dataset}")
        except Exception as e:
            print(f"Error loading dataset {dataset}: {str(e)}. Skipping.")
            continue
        
        # Find all fine-tuned models for this dataset
        model_dirs = glob.glob(os.path.join(args.input_dir, f"*_{dataset}_*"))
        if not model_dirs:
            print(f"No fine-tuned models found for {dataset}. Skipping.")
            continue
            
        for model_name, model_config in model_cfgs.items():
            # Check if results already exist for this model and dataset
            output_file = os.path.join(args.output_dir, f"{dataset}_{model_name}_accuracy.json")
            if os.path.exists(output_file):
                print(f"Accuracy results for {model_name} on {dataset} already exist at {output_file}. Skipping.")
                continue
                
            # Get model ID from config
            model_id = model_config["hf_id"]
            
            # Initialize model with vLLM for efficient generation
            try:
                print(f"Loading model: {model_id}")
                llm = LLM(
                    model=model_id,
                    dtype="float16",
                    tensor_parallel_size=model_config.get("tp", 1),
                    trust_remote_code=model_config.get("needs_trust", False)
                )
                
                # Set sampling parameters for generation
                sampling_params = SamplingParams(
                    temperature=0.0,  # greedy decoding
                    max_tokens=512,
                    stop=["Question:", "\n\n"],  # Stop generation at these tokens
                )
                
                results = []
                correct_count = 0
                
                # Process each example
                for i, example in enumerate(tqdm(ds, desc=f"Evaluating {model_name} on {dataset}")):
                    # Prepare prompt based on dataset type
                    if dataset == "aqua_rat":
                        question = example["question"]
                        options = example["options"]
                        correct = example["correct"]
                        prompt = build_prompt(question, options)
                    elif dataset == "arc_easy" or dataset == "arc":
                        question = example["question"]
                        options = "\n".join([f"{k}: {v}" for k, v in zip(example["choices"]["label"], example["choices"]["text"])])
                        correct = example["answerKey"]
                        prompt = build_prompt(question, options)
                    elif dataset == "gsm8k":
                        question = example["question"]
                        correct = example["answer"]
                        prompt = build_prompt(question)
                    elif dataset == "mmlu" or dataset == "mmlu_math":
                        question = example["question"]
                        options = "\n".join([f"{chr(65+i)}: {opt}" for i, opt in enumerate(example["choices"])])
                        correct = chr(65 + example["answer"])
                        prompt = build_prompt(question, options)
                    elif dataset == "cqna":
                        question = example["question"]
                        options = "\n".join([f"{label}: {text}" for label, text in zip(example["choices"]["label"], example["choices"]["text"])])
                        correct = example["answerKey"]
                        prompt = build_prompt(question, options)
                    else:
                        continue
                        
                    # Generate new chain-of-thought with model
                    outputs = llm.generate([prompt], sampling_params)
                    cot = outputs[0].outputs[0].text.strip()
                    
                    # Extract the predicted answer
                    predicted = extract_answer(cot)
                    
                    # Check if prediction is correct
                    is_correct = False
                    if predicted:
                        # Handle different formats of correct answers
                        if dataset == "aqua_rat" or dataset == "gsm8k":
                            # These answers might be indices or actual values
                            try:
                                if predicted.isdigit() and str(predicted) in str(correct):
                                    is_correct = True
                                elif predicted == correct:
                                    is_correct = True
                            except ValueError:
                                pass
                        else:  # ARC, MMLU, CQNA - letter-based answers
                            is_correct = (predicted.upper() == correct.upper())
                    
                    if is_correct:
                        correct_count += 1
                    
                    # Store result
                    result = {
                        "id": i,
                        "question": question,
                        "options": options if "options" in locals() else None,
                        "prompt": prompt,
                        "cot": cot,
                        "predicted": predicted,
                        "correct": correct,
                        "is_correct": is_correct
                    }
                    results.append(result)
                
                # Calculate accuracy
                accuracy = correct_count / len(results) if results else 0
                
                # Save detailed results to file
                with open(output_file, "w") as f:
                    json.dump({
                        "model": model_id,
                        "model_name": model_name,
                        "dataset": dataset,
                        "accuracy": accuracy,
                        "correct_count": correct_count,
                        "total_count": len(results),
                        "results": results
                    }, f, indent=2)
                
                # Cache predictions for future use in other evaluation metrics
                cache_file = os.path.join(args.input_dir, f"{dataset}_{model_name}_predictions.pkl")
                torch.save({
                    "model_id": model_id,
                    "model_name": model_name,
                    "dataset": dataset,
                    "results": results
                }, cache_file)
                
                print(f"Accuracy: {accuracy:.4f} ({correct_count}/{len(results)})")
                print(f"Detailed results saved to {output_file}")
                print(f"Prediction cache saved to {cache_file}")
                
                # Clean up GPU memory
                try:
                    del llm
                    torch.cuda.empty_cache()
                except:
                    pass
                    
            except Exception as e:
                print(f"Error evaluating {model_name} on {dataset}: {str(e)}. Skipping.")
                continue

if __name__ == "__main__":
    main()
