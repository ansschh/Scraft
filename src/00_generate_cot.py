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
    """Build a CoT prompt for the given question and options."""
    prompt = "Question: " + question + "\n"
    if options:
        # Handle both string and list formats for options
        if isinstance(options, list):
            options_text = "\n".join(options)
        else:
            options_text = options
        prompt += "Options:\n" + options_text + "\n"
    
    # Add explicit chain-of-thought instruction
    prompt += "\nLet's solve this step-by-step:\n1. "
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
        temperature=0.7,  # higher temperature for more diverse reasoning
        top_p=0.95,  # nucleus sampling
        max_tokens=2048,  # allow longer responses for complete reasoning
        stop=["Question:", "\n\n\n"],  # Less aggressive stopping
    )
    
    # Process each model
    for model_name, cfg in model_cfgs.items():
        # Make sure we use consistent naming - use the name from config or fallback to key
        name = cfg.get("name", model_name)
        output_file = os.path.join(args.output_dir, f"{name}_{args.dataset}_cot.json")
        
        # Delete output file if it exists to ensure a clean run
        if os.path.exists(output_file):
            print(f"Removing existing output file {output_file} for clean run...")
            os.remove(output_file)
        
        # Also remove any intermediate results with the same prefix (from previous runs)
        intermediate_pattern = f"{output_file.replace('.json', '')}*"
        import glob
        for intermediate_file in glob.glob(intermediate_pattern):
            if intermediate_file != output_file and os.path.exists(intermediate_file):
                print(f"Removing intermediate result file: {intermediate_file}")
                os.remove(intermediate_file)
            
        print(f"Processing {name} with clean output state...")
        
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
            
            # Store the result
            result = {
                "id": i,
                "question": question,
                "options": options,
                "cot": cot,
                "correct": correct
            }
            results.append(result)
            
            # Save intermediate results after EVERY example to avoid data loss
            try:
                # First write to a temporary file to prevent corruption
                temp_output_file = f"{output_file}.tmp"
                with open(temp_output_file, "w") as f:
                    json.dump(results, f, indent=2)
                
                # Then rename to the actual output file (atomic operation)
                import shutil
                shutil.move(temp_output_file, output_file)
                
                print(f"Saved intermediate results to {output_file} after example {i}")
            except Exception as save_error:
                print(f"Error saving results: {save_error}")
                # Try to save to an emergency backup file
                try:
                    backup_file = f"{output_file}.backup.{i}.json"
                    with open(backup_file, "w") as f:
                        json.dump(results, f, indent=2)
                    print(f"Saved emergency backup to {backup_file}")
                except:
                    print("CRITICAL: Failed to save backup!")
                    pass
        
        print(f"Generated CoT for {len(results)} examples with {name}, saved to {output_file}")
        
        # Free GPU memory
        del llm
        torch.cuda.empty_cache()
    

if __name__ == "__main__":
    main()
