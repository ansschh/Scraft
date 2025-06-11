#!/usr/bin/env python3
"""
Simple evaluator for SCRaFT CoT outputs that doesn't require model loading.
Extracts final answer letters/numbers from saved CoT files and compares to ground truth.
"""

import json
import os
import re
import argparse
from tqdm import tqdm
from datasets import load_dataset
import numpy as np

def extract_answer(cot_text):
    """Extract the final answer letter or number from the chain of thought."""
    # First, try to find explicit "The answer is X" pattern
    patterns = [
        r"(?:final answer is|the answer is|answer is|answer:|I choose)\s*(?:option)?\s*([A-E0-9])",
        r"(?:final answer is|the answer is|answer is|answer:|I choose)\s*\(([A-E0-9])\)",
        r"(?:final answer is|the answer is|answer is|answer:|I choose)\s*(?:option)?\s*([A-E])\.",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, cot_text, re.IGNORECASE)
        if match:
            return match.group(1)
            
    # Fall back to the last occurrence of a letter with a period or parenthesis
    letter_matches = re.findall(r'([A-E0-9])[.)]', cot_text)
    if letter_matches:
        return letter_matches[-1]
    
    # Fall back to the last standalone capital letter
    letter_matches = re.findall(r'\b([A-E])\b', cot_text)
    if letter_matches:
        return letter_matches[-1]
    
    # For numeric answers (e.g., GSM8K), try to find the last number
    numeric_match = re.search(r'answer\s*(?:is|=|:)?\s*(-?\d+(?:\.\d+)?)', cot_text, re.IGNORECASE)
    if numeric_match:
        return numeric_match.group(1)
        
    # Fallback: Return None if no answer found
    return None

def main():
    parser = argparse.ArgumentParser(description="Evaluate CoT accuracy without model loading")
    parser.add_argument("--model_list", type=str, required=True, help="Path to models.json config file")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing CoT JSON files")
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
    
    # Load model configurations
    with open(args.model_list, 'r') as f:
        model_cfgs = json.load(f)
    
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
            elif dataset.startswith("mmlu"):
                if dataset == "mmlu_math":
                    ds = load_dataset("cais/mmlu", "high_school_mathematics", split="test").shuffle(seed=42)[:args.samples]
                else:
                    ds = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=42)[:args.samples]
            else:
                print(f"Unknown dataset: {dataset}. Skipping.")
                continue
        except Exception as e:
            print(f"Error loading dataset {dataset}: {str(e)}. Skipping.")
            continue
        
        # Find all CoT files for this dataset - check multiple patterns
        cot_files = []
        for file in os.listdir(args.input_dir):
            model_name = None
            for model_key in model_cfgs.keys():
                if file.startswith(model_key) and (dataset in file):
                    model_name = model_key
                    cot_files.append((model_name, os.path.join(args.input_dir, file)))
                    break
        
        print(f"Found {len(cot_files)} CoT files for {dataset}: {[f[0] for f in cot_files]}")
        if not cot_files:
            print(f"No CoT files found for {dataset}. Skipping.")
            continue
            
        for model_name, cot_file_path in cot_files:
            # Check if results already exist for this model and dataset
            output_file = os.path.join(args.output_dir, f"{dataset}_{model_name}_accuracy.json")
            if os.path.exists(output_file):
                print(f"Accuracy results for {model_name} on {dataset} already exist at {output_file}. Skipping.")
                continue
            
            print(f"Evaluating {model_name} on {dataset} using saved CoT from {cot_file_path}")
            
            # Load CoT outputs
            try:
                with open(cot_file_path, 'r') as f:
                    cot_outputs = json.load(f)
            except Exception as e:
                print(f"Error loading CoT file {cot_file_path}: {e}")
                continue
                
            results = []
            correct_count = 0
            total_count = 0
                
            # Evaluate each example
            for i, example in enumerate(tqdm(ds, desc=f"Evaluating {model_name} on {dataset}")):
                if i >= len(cot_outputs):
                    print(f"Warning: Not enough CoT outputs for all examples. Stopping at {i}/{len(ds)}")
                    break
                    
                # Get the correct answer from the dataset
                if dataset == "aqua_rat":
                    question = example["question"]
                    options = example["options"]
                    correct = example["correct"]
                elif dataset == "arc_easy" or dataset == "arc":
                    question = example["question"]
                    correct = example["answerKey"]
                elif dataset == "gsm8k":
                    question = example["question"]
                    correct = example["answer"]
                elif dataset == "mmlu" or dataset == "mmlu_math":
                    question = example["question"]
                    correct = chr(65 + example["answer"])  # Convert to A, B, C, D
                else:
                    continue
                
                # Get the corresponding CoT for this example
                cot = cot_outputs[i]["output"]
                if not cot or cot == "[Error: Failed to generate reasoning]":
                    print(f"Warning: Missing or error in CoT for example {i}")
                    results.append({
                        "id": i,
                        "question": question,
                        "correct": correct,
                        "predicted": None,
                        "is_correct": False,
                        "cot": cot
                    })
                    total_count += 1
                    continue
                    
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
                    else:  # ARC, MMLU - letter-based answers
                        is_correct = (predicted.upper() == correct.upper())
                
                if is_correct:
                    correct_count += 1
                
                total_count += 1
                
                # Store result
                result = {
                    "id": i,
                    "question": question,
                    "correct": correct,
                    "predicted": predicted,
                    "is_correct": is_correct,
                    "cot": cot
                }
                results.append(result)
                
            # Calculate accuracy
            if total_count > 0:
                accuracy = correct_count / total_count
            else:
                accuracy = 0.0
                
            # Save results
            output_data = {
                "model": model_name,
                "dataset": dataset,
                "accuracy": accuracy,
                "correct_count": correct_count,
                "total_count": total_count,
                "results": results
            }
            
            with open(output_file, "w") as f:
                json.dump(output_data, f, indent=2)
                
            print(f"{model_name} on {dataset}: Accuracy = {accuracy:.4f} ({correct_count}/{total_count})")
    
    print("Evaluation complete. Results saved to", args.output_dir)

if __name__ == "__main__":
    main()
