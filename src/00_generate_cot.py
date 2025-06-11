#!/usr/bin/env python3
"""
Generates vanilla chain-of-thought reasoning for each dataset example.
Uses either vLLM or Hugging Face Transformers for token generation.
"""

# Set multiprocessing start method to 'spawn' to avoid CUDA initialization issues
import multiprocessing
import torch
import json
import os
from datasets import load_dataset
from tqdm import tqdm
import argparse

# Import our customized SamplingParams replacement for compatibility
class SamplingParams:
    def __init__(self, temperature=0.7, top_p=1.0, max_tokens=512, stop=None):
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.stop = stop

# This must be called before any multiprocessing operations
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

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
            try:
                if args.dataset == "aqua_rat":
                    # Handle both string and dict formats for datasets
                    if isinstance(example, dict):
                        question = example["question"]
                        options = example["options"]
                        correct = example["correct"]
                    else:
                        # Parse the example if it's a string
                        example_dict = json.loads(example) if isinstance(example, str) else example
                        question = example_dict["question"]
                        options = example_dict["options"]
                        correct = example_dict["correct"]
                    prompt = build_prompt(question, options)
                else:  # arc_easy
                    if isinstance(example, dict):
                        question = example["question"]
                        if isinstance(example["choices"], dict):
                            options = "\n".join([f"{k}: {v}" for k, v in zip(example["choices"]["label"], example["choices"]["text"])])
                        else:
                            # Handle different dataset formats
                            options = "\n".join([f"{c['label']}: {c['text']}" for c in example["choices"]])
                        correct = example["answerKey"]
                    else:
                        # Parse the example if it's a string
                        example_dict = json.loads(example) if isinstance(example, str) else example
                        question = example_dict["question"]
                        if isinstance(example_dict["choices"], dict):
                            options = "\n".join([f"{k}: {v}" for k, v in zip(example_dict["choices"]["label"], example_dict["choices"]["text"])])
                        else:
                            options = "\n".join([f"{c['label']}: {c['text']}" for c in example_dict["choices"]])
                        correct = example_dict["answerKey"]
                    prompt = build_prompt(question, options)
            except Exception as e:
                print(f"Error processing example {i}: {e}")
                print(f"Example content: {example}")
                # Use a simple fallback prompt
                if isinstance(example, str):
                    prompt = build_prompt(example)
                else:
                    prompt = build_prompt(str(example))
                
            # Generate CoT
            outputs = llm.generate([prompt], sampling_params)
            
            # Extract text from output (handle both vLLM and HuggingFace wrapper formats)
            if hasattr(outputs[0], 'outputs') and hasattr(outputs[0].outputs[0], 'text'):
                # vLLM format
                cot = outputs[0].outputs[0].text.strip()
            else:
                # HuggingFace wrapper format
                cot = outputs[0]['output'].strip()
            
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
