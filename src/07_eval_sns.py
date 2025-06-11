#!/usr/bin/env python3
"""
Sentence-Necessity Search (SNS) - greedy deletion of sentences until the model's answer changes.
Identifies the minimal set of necessary sentences in the chain of thought.
"""

import torch
import json
import os
import re
import argparse
import nltk
import copy
from tqdm import tqdm
from vllm import LLM, SamplingParams

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

def sentence_necessity_search(llm, prompt, original_chain):
    """
    Performs greedy deletion of sentences to find the minimal necessary set.
    Returns the necessary sentences and the sparsity ratio.
    """
    sampling_params = SamplingParams(
        temperature=0.0,  # greedy decoding
        max_tokens=512,
        stop=["Question:", "\n\n"],  # Stop generation at these tokens
    )
    
    # Extract original answer
    original_answer = extract_answer(original_chain)
    if not original_answer:
        return [], 0, []  # Can't evaluate if no clear answer
    
    # Split into sentences using NLTK's punkt tokenizer
    sentences = nltk.sent_tokenize(original_chain)
    if len(sentences) <= 1:
        return sentences, 1.0, []  # If only one sentence, it's necessary
    
    necessary_indices = []
    removal_history = []
    
    # Try removing each sentence one by one
    for i in range(len(sentences)):
        # Create a new chain without the current sentence
        test_sentences = sentences.copy()
        removed_sentence = test_sentences.pop(i)
        test_chain = " ".join(test_sentences)
        
        # Run the model on the modified chain
        test_prompt = prompt + "\n\n" + test_chain
        outputs = llm.generate([test_prompt], sampling_params)
        new_chain = outputs[0].outputs[0].text.strip()
        new_answer = extract_answer(new_chain)
        
        # Check if removing this sentence changes the answer
        answer_changed = (new_answer != original_answer) or not new_answer
        
        # Record the result
        removal_history.append({
            "index": i,
            "removed_sentence": removed_sentence,
            "new_chain": new_chain,
            "new_answer": new_answer,
            "answer_changed": answer_changed
        })
        
        # If removing the sentence changes the answer, it's necessary
        if answer_changed:
            necessary_indices.append(i)
    
    # Calculate sparsity ratio (proportion of necessary sentences)
    sparsity_ratio = len(necessary_indices) / len(sentences) if sentences else 0
    
    # Get the list of necessary sentences
    necessary_sentences = [sentences[i] for i in necessary_indices]
    
    return necessary_sentences, sparsity_ratio, removal_history

def main():
    parser = argparse.ArgumentParser(description="Evaluate sentence necessity")
    parser.add_argument("--model_list", type=str, required=True, help="Path to models.json config file")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing prediction caches")
    parser.add_argument("--output_dir", type=str, default="logs/sns", help="Directory to save results")
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
    
    # Ensure NLTK punkt tokenizer is available
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
                
            # Check if SNS results already exist for this model and dataset
            output_file = os.path.join(args.output_dir, f"{dataset}_{model_name}_sns.json")
            if os.path.exists(output_file):
                print(f"SNS results for {model_name} on {dataset} already exist at {output_file}. Skipping.")
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
                print(f"Processing SNS evaluation for {model_name} ({model_id}) on {dataset}")
                
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
                sns_results = []
                total_sparsity = 0
                
                for i, result in enumerate(tqdm(results, desc=f"Evaluating {model_name} on {dataset}")):
                    prompt = result["prompt"]
                    original_chain = result["cot"]
                    
                    # Perform sentence necessity search
                    necessary_sentences, sparsity_ratio, removal_history = sentence_necessity_search(
                        llm, prompt, original_chain
                    )
                    
                    total_sparsity += sparsity_ratio
                    
                    # Store SNS result
                    sns_result = copy.deepcopy(result)
                    sns_result.update({
                        "necessary_sentences": necessary_sentences,
                        "sparsity_ratio": sparsity_ratio,
                        "sentence_count": len(nltk.sent_tokenize(original_chain)),
                        "necessary_count": len(necessary_sentences),
                        "removal_history": removal_history
                    })
                    sns_results.append(sns_result)
                
                # Calculate overall sparsity ratio
                avg_sparsity = total_sparsity / len(sns_results) if sns_results else 0
                
                # Save detailed results to file
                with open(output_file, "w") as f:
                    json.dump({
                        "model": model_id,
                        "model_name": model_name,
                        "dataset": dataset,
                        "average_sparsity_ratio": avg_sparsity,
                        "results": sns_results
                    }, f, indent=2)
                
                # Also save a summary file for quick reference
                summary_file = os.path.join(args.output_dir, f"{dataset}_{model_name}_sns_summary.json")
                with open(summary_file, "w") as f:
                    json.dump({
                        "model": model_id,
                        "model_name": model_name,
                        "dataset": dataset,
                        "average_sparsity_ratio": avg_sparsity
                    }, f, indent=2)
                
                print(f"Average Sparsity Ratio: {avg_sparsity:.4f}")
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
