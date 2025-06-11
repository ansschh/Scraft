#!/usr/bin/env python3
"""
Generates vanilla chain-of-thought reasoning for each dataset example.
Uses vLLM for efficient token generation with paged attention.
"""

import torch
import json
import os
from datasets import load_dataset
from vllm import LLM, SamplingParams
from tqdm import tqdm
import argparse

# Build prompt template - keeping CoT start marker outside user text
def build_prompt(question, options=None):
    if options:
        prompt = f"""Question: {question}

Options: {options}

Think step-by-step to solve this problem."""
    else:
        prompt = f"""Question: {question}

Think step-by-step to solve this problem."""
    return prompt

def main():
    parser = argparse.ArgumentParser(description="Generate CoT explanations")
    parser.add_argument("--model_list", type=str, required=True, help="Path to configs/models.json")
    parser.add_argument("--dataset", type=str, choices=["aqua_rat", "arc_easy"], required=True, help="Dataset to use")
    parser.add_argument("--samples", type=int, default=500, help="Number of samples to process")
    parser.add_argument("--output_dir", type=str, default="logs", help="Directory to save results")
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(0)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    if args.dataset == "aqua_rat":
        ds = load_dataset("aqua_rat", split="test").shuffle(seed=42)[:args.samples]
    elif args.dataset == "arc_easy":
        ds = load_dataset("ai2_arc", "ARC-Easy", split="test").shuffle(seed=42)[:args.samples]
    
    # Load model configurations
    from utils import load_model
    import json as j
    model_cfgs = j.load(open(args.model_list))
    
    # Set sampling parameters for generation
    sampling_params = SamplingParams(
        temperature=0.0,  # greedy decoding
        max_tokens=512,
        stop=["Question:", "\n\n"],  # Stop generation at these tokens
    )
    
    # Process each model
    for name, cfg in model_cfgs.items():
        # Set output path for this model
        out_path = os.path.join(args.output_dir, f"{name}_{args.dataset}_cot.json")
        
        # Skip if already processed
        if os.path.exists(out_path):
            print(f"Skipping {name} (output exists)")
            continue
            
        print(f"Processing {name}...")
        
        # Load model with vLLM
        llm = load_model(cfg)
        
        results = []
        
        # Process each example
        for i, example in enumerate(tqdm(ds, desc=f"Model: {name}")):
            if args.dataset == "aqua_rat":
                question = example["question"]
                options = example["options"]
                correct = example["correct"]
                prompt = build_prompt(question, options)
            else:  # arc_easy
                question = example["question"]
                options = "\n".join([f"{k}: {v}" for k, v in zip(example["choices"]["label"], example["choices"]["text"])])
                correct = example["answerKey"]
                prompt = build_prompt(question, options)
                
            # Generate CoT
            outputs = llm.generate([prompt], sampling_params)
            cot = outputs[0].outputs[0].text.strip()
            
            # Store result
            result = {
                "id": i,
                "question": question,
                "options": options,
                "correct": correct,
                "prompt": prompt,
                "cot": cot
            }
            results.append(result)
        
        # Save results to file for this model
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Generated CoT for {len(results)} examples with {name}, saved to {out_path}")
        
        # Free GPU memory
        del llm
        torch.cuda.empty_cache()
    

if __name__ == "__main__":
    main()
