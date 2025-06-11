#!/usr/bin/env python3
"""
Generates vanilla chain-of-thought reasoning for each dataset example.
Uses Hugging Face Transformers for dataset loading and token generation.
"""

# Set multiprocessing start method to 'spawn' to avoid CUDA initialization issues
import multiprocessing
import torch
import json
import os
from datasets import load_dataset
from tqdm import tqdm
import argparse

# Define a simple Generation Configuration class that's compatible with our code
from transformers import GenerationConfig

class SamplingParams:
    """Simple wrapper for generation parameters"""
    def __init__(self, temperature=0.0, top_p=1.0, max_tokens=512, stop=None):
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.stop = stop
        
    def to_generation_config(self):
        """Convert to HF GenerationConfig"""
        return GenerationConfig(
            max_new_tokens=self.max_tokens,
            temperature=self.temperature if self.temperature > 0 else 1.0,
            do_sample=self.temperature > 0,
            top_p=self.top_p,
            # We'll handle stop tokens separately
        )

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
    
    # Load dataset with explicit debug output
    print(f"Loading dataset: {args.dataset}")
    
    try:
        if args.dataset == "aqua_rat":
            print("Loading aqua_rat from Hugging Face...")
            ds = load_dataset("aqua_rat", split="validation")
            print(f"Dataset loaded successfully. Length: {len(ds)}")
            # Print first example for debugging
            if len(ds) > 0:
                print(f"Example structure: {ds[0]}")
        else:  # arc_easy
            print("Loading ai2_arc/ARC-Easy from Hugging Face...")
            ds = load_dataset("ai2_arc", "ARC-Easy", split="test")
            print(f"Dataset loaded successfully. Length: {len(ds)}")
            # Print first example for debugging
            if len(ds) > 0:
                print(f"Example structure: {ds[0]}")
    except Exception as e:
        print(f"Error loading dataset from Hugging Face: {e}")
        raise
    
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
        
        # Load model with Hugging Face through our utils function
        llm = load_model(cfg)
        
        results = []
        
        # Process each example
        for i, example in enumerate(tqdm(ds, desc=f"Model: {name}")):
            # Extract data from Hugging Face datasets (already properly structured)
            if args.dataset == "aqua_rat":
                question = example["question"]
                options = example["options"]
                correct = example["correct"]
                prompt = build_prompt(question, options)
                
            else:  # arc_easy
                question = example["question"]
                # Format choices consistently
                options = "\n".join([f"{label}: {text}" 
                                  for label, text in zip(example["choices"]["label"], 
                                                       example["choices"]["text"])])
                correct = example["answerKey"]
                prompt = build_prompt(question, options)
                
            print(f"Generating for example {i}...")
            # Generate CoT using the HuggingFaceLLMWrapper
            print(f"Generating for example {i}...")
            try:
                outputs = llm.generate([prompt], sampling_params)
                # Extract text from output using HuggingFace wrapper format
                cot = outputs[0]['output'].strip()
                print(f"Generation complete for example {i}")
            except Exception as e:
                print(f"Generation error for example {i}: {e}")
                cot = "Error in generation"
            
            # Store result
            result = {
                "id": i,
                "question": question,
                "options": options,
                "cot": cot,
                "correct": correct
            }
            print(f"Result for example {i} created")
            results.append(result)
            # Save intermediate results
            print(f"Saving intermediate results to {out_path}")
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
        
        print(f"Generated CoT for {len(results)} examples with {name}, saved to {out_path}")
        
        # Free GPU memory
        del llm
        torch.cuda.empty_cache()
    

if __name__ == "__main__":
    main()
