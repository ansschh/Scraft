#!/usr/bin/env python3
"""
Prompts the same model to repair its own contradictions.
This creates positive examples for fine-tuning without human labels.
"""

import torch
import json
import os
from vllm import LLM, SamplingParams
from tqdm import tqdm
import argparse

def build_refine_prompt(question, options, contradicted_cot):
    """
    Build a prompt that asks the model to fix the contradictory chain-of-thought.
    The contradicted chain is echoed verbatim before instructions to keep it in context.
    """
    prompt = f"""Question: {question}

Options: {options}

Chain of thought with a logical error:
{contradicted_cot}

The chain of thought above contains exactly one logical error or contradiction.
Please identify the error and provide a corrected chain of thought.
Return only the revised chain-of-thought."""
    return prompt

def main():
    parser = argparse.ArgumentParser(description="Self-refine contradicted chains")
    parser.add_argument("--model_list", type=str, required=True, help="Path to configs/models.json")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with contradicted CoT JSON files")
    parser.add_argument("--output_dir", type=str, default="logs", help="Directory to save results")
    parser.add_argument("--datasets", nargs="+", default=["aqua_rat", "arc_easy"], help="Datasets to process")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    torch.manual_seed(0)
    
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
    
    # Process each dataset
    for dataset in args.datasets:
        # Find all contradicted CoT JSON files for this dataset
        input_files = []
        for file in os.listdir(args.input_dir):
            if file.endswith(f"_{dataset}_contradicted.json"):
                input_files.append(os.path.join(args.input_dir, file))
        
        if not input_files:
            print(f"No contradicted files found for {dataset}")
            continue
        
        print(f"Found {len(input_files)} contradicted files for {dataset}")
        
        # Process each model file
        for input_file in input_files:
            model_name = os.path.basename(input_file).split(f"_{dataset}")[0]
            output_file = os.path.join(args.output_dir, f"{model_name}_{dataset}_repaired.json")
            
            # Skip if already processed
            if os.path.exists(output_file):
                print(f"Skipping {model_name} (output exists)")
                continue
            
            # We only use Llama-3-8B for self-refinement as in the original paper
            refiner_model = "meta-llama/Meta-Llama-3-8B-Instruct"
            print(f"Processing {model_name} using {refiner_model} for refinement...")
            
            # Load data
            with open(input_file, "r") as f:
                data = json.load(f)
            
            # Initialize the refiner model with vLLM
            llm = load_model({"id": refiner_model})
            
            # Process each example
            for item in tqdm(data, desc=f"Refining {model_name}"):
                question = item["question"]
                options = item["options"]
                contradicted_cot = item["contradicted_cot"]
                
                # Build refine prompt
                prompt = build_refine_prompt(question, options, contradicted_cot)
                
                # Generate repaired CoT
                outputs = llm.generate([prompt], sampling_params)
                repaired_cot = outputs[0].outputs[0].text.strip()
                
                # Store result
                item["refine_prompt"] = prompt
                item["repaired_cot"] = repaired_cot
            
            # Save results
            with open(output_file, "w") as f:
                json.dump(data, f, indent=2)
            
            print(f"Self-refined {len(data)} contradicted examples with {model_name}, saved to {output_file}")
            
            # Free GPU memory
            del llm
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
