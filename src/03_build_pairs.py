#!/usr/bin/env python3
"""
Converts (prompt, good, bad) triples into DPO preference pairs.
Appends the same answer token to both chosen and rejected branches to
isolate optimization to the process tokens only.
"""

import json
import os
import argparse
import re
from tqdm import tqdm
from datasets import Dataset

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

def build_dpo_pairs(data):
    """Build DPO preference pairs from the data."""
    pairs = []
    for item in tqdm(data):
        question = item["question"]
        options = item["options"] 
        
        # Extract the answer from both original and repaired chains
        original_answer = extract_answer(item["cot"])
        repaired_answer = extract_answer(item["repaired_cot"])
        
        # Check if the answer is the same (it should be)
        if original_answer == repaired_answer:
            answer = original_answer
        else:
            # If different, prioritize the repaired answer (should be rare)
            answer = repaired_answer
        
        # Build input prompt
        input_prompt = f"Question: {question}\n\nOptions: {options}\n\nThink step-by-step to solve this problem."
        
        # Build chosen and rejected outputs
        # Both have the same answer but different chains of thought
        chosen_output = item["repaired_cot"] + f"\n\nAnswer: {answer}"
        rejected_output = item["contradicted_cot"] + f"\n\nAnswer: {answer}"
        
        pair = {
            "input": input_prompt,
            "chosen": chosen_output,
            "rejected": rejected_output,
            "original_answer": original_answer,
            "repaired_answer": repaired_answer,
            "id": item.get("id", len(pairs)),
        }
        pairs.append(pair)
    
    return pairs

def main():
    parser = argparse.ArgumentParser(description="Build DPO preference pairs")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with repaired CoT JSON files")
    parser.add_argument("--output_dir", type=str, default="logs", help="Directory to save results")
    parser.add_argument("--datasets", nargs="+", default=["aqua_rat", "arc_easy"], help="Datasets to process")
    parser.add_argument("--model_size", type=str, default="8B", help="Model size to use for fine-tuning (e.g., 8B)")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # We'll collect all pairs from different datasets into one combined training set
    all_pairs = []
    
    # Process each dataset
    for dataset in args.datasets:
        # Find the specific model file we want to use for SCRaFT fine-tuning
        # We use the specified model size (default 8B) in the original SCRaFT paper
        target_files = []
        for file in os.listdir(args.input_dir):
            if file.endswith(f"_{dataset}_repaired.json") and args.model_size in file:
                target_files.append(os.path.join(args.input_dir, file))
        
        if not target_files:
            print(f"No repaired files found for {dataset} with model size {args.model_size}")
            continue
            
        if len(target_files) > 1:
            print(f"Warning: Multiple matching files found for {dataset} with model size {args.model_size}")
            print(f"Using: {target_files[0]}")
        
        input_file = target_files[0]
        model_name = os.path.basename(input_file).split(f"_{dataset}")[0]
        
        print(f"Building DPO pairs for {model_name} on {dataset}...")
        
        # Load data
        with open(input_file, "r") as f:
            data = json.load(f)
        
        # Build DPO preference pairs
        pairs = build_dpo_pairs(data)
        print(f"Generated {len(pairs)} pairs for {dataset}")
        
        # Add dataset name to pairs metadata
        for pair in pairs:
            pair["dataset"] = dataset
        
        # Add to combined dataset
        all_pairs.extend(pairs)
        
        # Save individual dataset results as JSON
        output_file = os.path.join(args.output_dir, f"{model_name}_{dataset}_dpo_pairs.json")
        with open(output_file, "w") as f:
            json.dump(pairs, f, indent=2)
    
    # Skip further processing if we didn't find any pairs
    if not all_pairs:
        print("No pairs were generated. Check input files and model size parameter.")
        return
    
    # Save combined dataset
    combined_output = os.path.join(args.output_dir, f"combined_{args.model_size}_dpo_pairs.json")
    with open(combined_output, "w") as f:
        json.dump(all_pairs, f, indent=2)
    
    # Convert to HuggingFace Dataset for compatibility with DPOTrainer
    dataset = Dataset.from_list(all_pairs)
    dataset_path = os.path.join(args.output_dir, f"combined_{args.model_size}_dpo_dataset")
    dataset.save_to_disk(dataset_path)
    
    print(f"Built {len(all_pairs)} total DPO preference pairs:")
    print(f"- Combined JSON saved to: {combined_output}")
    print(f"- Combined Dataset saved to: {dataset_path}")

if __name__ == "__main__":
    main()
