#!/usr/bin/env python3
"""
Mutates each chain-of-thought by flipping exactly one fact to create 
a logical contradiction, creating synthetic negative examples.
"""

import re
import json
import os
import nltk
import random
import argparse
from tqdm import tqdm

# Download NLTK punkt tokenizer if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Compiled regex patterns with lambda functions that create contradictions
NEG = [
    # Arithmetic contradiction: changes the result but keeps operands the same
    (re.compile(r'(\d+)\s*\+\s*(\d+)\s*=\s*(\d+)'), 
     lambda m: f"{m.group(1)} + {m.group(2)} = {int(m.group(3)) + 1}"),
    
    # Multiplication contradiction
    (re.compile(r'(\d+)\s*\*\s*(\d+)\s*=\s*(\d+)'), 
     lambda m: f"{m.group(1)} * {m.group(2)} = {int(m.group(3)) + 1}"),
    
    # Division contradiction
    (re.compile(r'(\d+)\s*/\s*(\d+)\s*=\s*(\d+)'), 
     lambda m: f"{m.group(1)} / {m.group(2)} = {float(m.group(3)) + 0.5}"),
    
    # Greater/less than contradiction
    (re.compile(r'(\d+)\s*>\s*(\d+)'), lambda m: f"{m.group(1)} < {m.group(2)}"),
    (re.compile(r'(\d+)\s*<\s*(\d+)'), lambda m: f"{m.group(1)} > {m.group(2)}"),
    
    # Equal/not equal contradiction
    (re.compile(r'(\w+)\s+is\s+equal\s+to\s+(\w+)'), lambda m: f"{m.group(1)} is not equal to {m.group(2)}"),
    (re.compile(r'(\w+)\s+is\s+not\s+equal\s+to\s+(\w+)'), lambda m: f"{m.group(1)} is equal to {m.group(2)}"),
    
    # True/false contradiction
    (re.compile(r'([\w\s]+)\s+is\s+true'), lambda m: f"{m.group(1)} is false"),
    (re.compile(r'([\w\s]+)\s+is\s+false'), lambda m: f"{m.group(1)} is true"),
]

def inject_contradiction(cot):
    """Inject a contradiction into the chain-of-thought."""
    # Split the chain-of-thought into sentences
    sentences = nltk.sent_tokenize(cot)
    
    # If we have fewer than 2 sentences, use the fallback
    if len(sentences) < 2:
        return cot + " NOT TRUE: " + cot.split()[0] if cot else "NOT TRUE: Empty reasoning."
    
    # Try each sentence until we find one that can be contradicted
    for i, sentence in enumerate(sentences):
        for pattern, replacement_fn in NEG:
            match = pattern.search(sentence)
            if match:
                contradicted_sentence = pattern.sub(replacement_fn(match), sentence)
                # Create a new chain with the contradicted sentence
                sentences[i] = contradicted_sentence
                return " ".join(sentences)
    
    # Fallback: if no pattern matches, simply negate a random sentence
    i = random.randint(0, len(sentences) - 1)
    sentences[i] = "NOT TRUE: " + sentences[i]
    return " ".join(sentences)

def main():
    parser = argparse.ArgumentParser(description="Inject contradictions into CoT chains")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with CoT JSON files")
    parser.add_argument("--output_dir", type=str, default="logs", help="Directory to save results")
    parser.add_argument("--datasets", nargs="+", default=["aqua_rat", "arc_easy"], help="Datasets to process")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Find all CoT JSON files for each dataset
    for dataset in args.datasets:
        pattern = f"*_{dataset}_cot.json"
        input_files = []
        for file in os.listdir(args.input_dir):
            if file.endswith(f"_{dataset}_cot.json"):
                input_files.append(os.path.join(args.input_dir, file))
        
        if not input_files:
            print(f"No CoT files found for {dataset}")
            continue
            
        print(f"Processing {len(input_files)} files for {dataset}...")
        
        # Process each model file
        for input_file in input_files:
            model_name = os.path.basename(input_file).split(f"_{dataset}")[0]
            output_file = os.path.join(args.output_dir, f"{model_name}_{dataset}_contradicted.json")
            
            # Skip if already processed
            if os.path.exists(output_file):
                print(f"Skipping {model_name} (output exists)")
                continue
                
            print(f"Processing {model_name}...")
            
            # Load data
            with open(input_file, "r") as f:
                data = json.load(f)
            
            # Process each example
            for item in tqdm(data, desc=f"Model: {model_name}"):
                # Inject contradiction into the chain-of-thought
                item["contradicted_cot"] = inject_contradiction(item["cot"])
            
            # Save results
            with open(output_file, "w") as f:
                json.dump(data, f, indent=2)
            
            print(f"Injected contradictions for {len(data)} examples with {model_name}, saved to {output_file}")

if __name__ == "__main__":
    main()
