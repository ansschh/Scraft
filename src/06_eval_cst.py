#!/usr/bin/env python3
"""
Contradiction Sensitivity Test (CST) - repeats the mutation step k times
per example and checks if any mutation flips the answer.
Uses cached predictions to save compute by only re-evaluating the mutated chain.
"""

# Set multiprocessing start method to 'spawn' to avoid CUDA initialization issues
import multiprocessing
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

import torch
import json
import os
import re
import argparse
import random
import nltk
import copy
from tqdm import tqdm
from vllm import LLM, SamplingParams

# Import contradiction functions from the mutation script
import importlib
contradiction_module = importlib.import_module("src.01_inject_contradiction")
NEG = contradiction_module.NEG
inject_contradiction = contradiction_module.inject_contradiction

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
    
    # If no clear answer found, return the last token as fallback
    words = cot_text.strip().split()
    if words:
        return words[-1]
    
    return ""

def evaluate_chain_sensitivity(llm, prompt, original_chain, k=3):
    """
    Test if the model's answer changes when k contradictions are inserted.
    Returns sensitivity score and flipped answer (if any).
    """
    sampling_params = SamplingParams(
        temperature=0.0,  # greedy decoding
        max_tokens=512,
        stop=["Question:", "\n\n"],  # Stop generation at these tokens
    )
    
    # Extract original answer
    original_answer = extract_answer(original_chain)
    if not original_answer:
        return 0, None, []  # Can't evaluate sensitivity if no clear answer
    
    # Generate k different contradicted chains
    contradictions = []
    random.seed(42)  # For reproducibility
    
    for i in range(k):
        # Generate a new contradiction
        contradicted_chain = inject_contradiction(original_chain)
        
        # Run the model on the contradicted prompt to get new chain and answer
        full_prompt = prompt + "\n\n" + contradicted_chain
        outputs = llm.generate([full_prompt], sampling_params)
        new_chain = outputs[0].outputs[0].text.strip()
        new_answer = extract_answer(new_chain)
        
        # Check if the answer changed
        is_sensitive = (new_answer != original_answer) and new_answer
        contradictions.append({
            "contradicted_chain": contradicted_chain,
            "new_chain": new_chain,
            "new_answer": new_answer,
            "is_sensitive": is_sensitive
        })
    
    # Calculate sensitivity (1 if any contradiction caused answer to flip)
    sensitivity = any(c["is_sensitive"] for c in contradictions)
    # Return the first flipped answer (if any)
    flipped_answer = next((c["new_answer"] for c in contradictions if c["is_sensitive"]), None)
    
    return int(sensitivity), flipped_answer, contradictions

def main():
    parser = argparse.ArgumentParser(description="Evaluate contradiction sensitivity")
    parser.add_argument("--model_list", type=str, required=True, help="Path to models.json config file")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing prediction caches")
    parser.add_argument("--k", type=int, default=3, help="Number of contradictions to test per example")
    parser.add_argument("--output_dir", type=str, default="logs/cst", help="Directory to save results")
    parser.add_argument("--datasets", nargs="+", default=["aqua_rat", "arc_easy"], help="Datasets to evaluate")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model configurations
    from utils import load_model
    import json as j
    import glob
    model_cfgs = j.load(open(args.model_list))
    
    # Set random seed for reproducibility
    torch.manual_seed(0)
    
    # Ensure NLTK punkt tokenizer is available (needed for sentence splitting)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    # Create a dictionary to cache models by name to avoid reloading
    model_cache = {}
    
    # Scan input directory for prediction caches
    # Example filename pattern: <dataset>_<model_name>_predictions.pkl
    for dataset in args.datasets:
        # Find all prediction files for this dataset
        prediction_files = glob.glob(os.path.join(args.input_dir, f"{dataset}_*_predictions.pkl"))
        
        for pred_file in prediction_files:
            # Extract model name from filename
            model_name = os.path.basename(pred_file).replace(f"{dataset}_", "").replace("_predictions.pkl", "")
            
            # Check if this model is in our config
            model_found = False
            for config_name, cfg in model_cfgs.items():
                if model_name in config_name:
                    model_cfg = cfg
                    model_found = True
                    break
            
            if not model_found:
                print(f"Skipping {pred_file}: Model {model_name} not found in config.")
                continue
                
            # Check if CST results already exist for this model and dataset
            output_file = os.path.join(args.output_dir, f"{dataset}_{model_name}_cst.json")
            if os.path.exists(output_file):
                print(f"CST results for {model_name} on {dataset} already exist at {output_file}. Skipping.")
                continue
                
            # Load cache file with predictions
            try:
                cache = torch.load(pred_file)
                results = cache.get("results", [])
                if not results:
                    print(f"No prediction results found in {pred_file}. Skipping.")
                    continue
                    
                # Get model ID from config
                model_id = model_cfg["hf_id"]
                print(f"Processing CST evaluation for {model_name} ({model_id}) on {dataset}")
                
                # Use cached model if available, otherwise initialize
                if model_name in model_cache:
                    print(f"Using cached model instance for {model_name}")
                    llm = model_cache[model_name]
                else:
                    # Initialize model with vLLM for efficient generation
                    try:
                        # Apply tensor parallel size if provided
                        tp_size = model_cfg.get("tp", 1)
                        llm = LLM(
                            model=model_id, 
                            dtype="float16", 
                            tensor_parallel_size=tp_size,
                            trust_remote_code=model_cfg.get("needs_trust", False),
                            quantization="awq" if model_cfg.get("load_in_4bit", False) else None
                        )
                        # Cache the model for reuse
                        model_cache[model_name] = llm
                    except Exception as e:
                        print(f"Failed to load model {model_id}: {str(e)}. Skipping.")
                        continue
                
                # Process each example
                cst_results = []
                sensitive_count = 0
                
                for i, result in enumerate(tqdm(results, desc=f"Evaluating {model_name} on {dataset}")):
                    prompt = result["prompt"]
                    original_chain = result["cot"]
                    
                    # Evaluate chain sensitivity
                    is_sensitive, flipped_answer, contradictions = evaluate_chain_sensitivity(
                        llm, prompt, original_chain, k=args.k
                    )
                    
                    if is_sensitive:
                        sensitive_count += 1
                    
                    # Store CST result
                    cst_result = copy.deepcopy(result)
                    cst_result.update({
                        "is_sensitive": bool(is_sensitive),
                        "sensitivity_score": is_sensitive,
                        "flipped_answer": flipped_answer,
                        "contradictions": contradictions
                    })
                    cst_results.append(cst_result)
                
                # Calculate overall sensitivity score
                sensitivity_rate = sensitive_count / len(cst_results) if cst_results else 0.0
                
                # Save detailed results to file
                with open(output_file, "w") as f:
                    json.dump({
                        "model": model_id,
                        "model_name": model_name,
                        "dataset": dataset,
                        "sensitivity_rate": sensitivity_rate,
                        "sensitive_count": sensitive_count,
                        "total_count": len(cst_results),
                        "k": args.k,
                        "results": cst_results
                    }, f, indent=2)
                
                # Also save a summary file for quick reference
                summary_file = os.path.join(args.output_dir, f"{dataset}_{model_name}_cst_summary.json")
                with open(summary_file, "w") as f:
                    json.dump({
                        "model": model_id,
                        "model_name": model_name,
                        "dataset": dataset,
                        "sensitivity_rate": sensitivity_rate,
                        "sensitive_count": sensitive_count,
                        "total_count": len(cst_results),
                        "k": args.k
                    }, f, indent=2)
                
                print(f"Contradiction Sensitivity Rate: {sensitivity_rate:.4f} ({sensitive_count}/{len(cst_results)})")
                print(f"Detailed results saved to {output_file}")
                print(f"Summary saved to {summary_file}")
                
                # Clean up model to free GPU memory
                del llm
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error processing {pred_file}: {str(e)}. Skipping.")
                continue

if __name__ == "__main__":
    main()
