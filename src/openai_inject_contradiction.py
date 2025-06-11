#!/usr/bin/env python3
"""
Improved contradiction injection using OpenAI API for the SCRaFT pipeline.
This script uses GPT to inject logical contradictions into chain-of-thought reasoning,
creating higher quality negative examples for training and evaluation.
"""

import os
import json
import argparse
from tqdm import tqdm
import time
from pathlib import Path
import openai
from openai import OpenAI

# Setup OpenAI client
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY", ""),
)

def inject_contradiction_with_gpt(question, options, cot_text):
    """
    Use GPT to inject a logical contradiction into the chain-of-thought.
    Returns the modified chain-of-thought with exactly one contradiction.
    """
    # If API key not set, fall back to regex method
    if not os.environ.get("OPENAI_API_KEY"):
        return fallback_inject_contradiction(cot_text)
    
    system_message = """
    You are an AI assistant that specializes in creating logical contradictions.
    Your task is to modify a given chain-of-thought reasoning by introducing EXACTLY ONE logical contradiction.
    
    Rules:
    1. Change EXACTLY ONE fact, calculation, or logical step to make it contradictory
    2. The contradiction should be subtle but clear enough to affect the final answer
    3. Do not change any other parts of the reasoning
    4. Preserve the overall structure and length of the reasoning
    5. The contradiction should be strictly logical, not just a change of opinion
    6. Return the FULL modified chain-of-thought with the contradiction
    
    Good contradictions:
    - Changing a mathematical operation result: "10 + 5 = 15" becomes "10 + 5 = 20"
    - Flipping a comparison: "x > y" becomes "x < y"
    - Negating a statement: "is possible" becomes "is impossible"
    """
    
    user_message = f"""
    QUESTION: {question}
    
    OPTIONS: {options}
    
    ORIGINAL CHAIN-OF-THOUGHT:
    {cot_text}
    
    Create a modified version with exactly ONE logical contradiction:
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", # Fast and cost-effective
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,  # Some creativity needed
            max_tokens=1024,  # Enough for full response
        )
        
        contradicted_cot = response.choices[0].message.content.strip()
        
        # Check if we got a reasonable response
        if len(contradicted_cot) < len(cot_text) * 0.5 or len(contradicted_cot) > len(cot_text) * 1.5:
            print("GPT response size differs too much from original. Using fallback.")
            return fallback_inject_contradiction(cot_text)
            
        return contradicted_cot
        
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return fallback_inject_contradiction(cot_text)

def fallback_inject_contradiction(cot_text):
    """
    Fallback method to inject a contradiction if the API call fails.
    """
    import random
    import re
    
    # Simple patterns to inject contradictions
    patterns = [
        # Addition contradiction
        (r'(\d+)\s*\+\s*(\d+)\s*=\s*(\d+)', 
         lambda m: f"{m.group(1)} + {m.group(2)} = {int(m.group(3)) + random.randint(1, 5)}"),
         
        # Multiplication contradiction
        (r'(\d+)\s*\*\s*(\d+)\s*=\s*(\d+)', 
         lambda m: f"{m.group(1)} * {m.group(2)} = {int(m.group(3)) + random.randint(1, 5)}"),
         
        # Division contradiction
        (r'(\d+)\s*/\s*(\d+)\s*=\s*(\d+)', 
         lambda m: f"{m.group(1)} / {m.group(2)} = {float(m.group(3)) + 0.5}"),
        
        # Greater/less than contradiction
        (r'(\d+)\s*>\s*(\d+)', lambda m: f"{m.group(1)} < {m.group(2)}"),
        (r'(\d+)\s*<\s*(\d+)', lambda m: f"{m.group(1)} > {m.group(2)}"),
        
        # Equal/not equal contradiction
        (r'(\w+)\s+is\s+equal\s+to\s+(\w+)', lambda m: f"{m.group(1)} is not equal to {m.group(2)}"),
        (r'(\w+)\s+is\s+not\s+equal\s+to\s+(\w+)', lambda m: f"{m.group(1)} is equal to {m.group(2)}"),
        
        # Yes/no contradiction
        (r'\b(Yes)\b', lambda m: "No"),
        (r'\b(No)\b', lambda m: "Yes"),
        
        # True/false contradiction
        (r'([\w\s]+)\s+is\s+true', lambda m: f"{m.group(1)} is false"),
        (r'([\w\s]+)\s+is\s+false', lambda m: f"{m.group(1)} is true"),
    ]
    
    sentences = cot_text.split('.')
    
    # Select a random sentence (excluding very short ones)
    valid_sentences = [s for s in sentences if len(s) > 10]
    if not valid_sentences:
        return cot_text + " [CONTRADICTION: This statement is false]"
        
    # Try each sentence until we find one that can be contradicted
    random.shuffle(valid_sentences)
    
    for sentence in valid_sentences:
        for pattern, replacement_fn in patterns:
            match = re.search(pattern, sentence)
            if match:
                contradicted_sentence = re.sub(pattern, replacement_fn(match), sentence)
                # Replace the original sentence with the contradicted one
                return cot_text.replace(sentence, contradicted_sentence)
    
    # Fallback: if no pattern matches, add a contradiction marker
    i = random.randint(0, len(sentences) - 1)
    if len(sentences[i]) > 5:
        contradiction = sentences[i] + " [CONTRADICTION: The opposite is true]"
        sentences[i] = contradiction
        return ".".join(sentences)
    else:
        return cot_text + " [CONTRADICTION: This reasoning contains an error]"

def process_file(input_file, output_file):
    """
    Process a single CoT file, inject contradictions, and save to output file.
    """
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
            
        # Make a copy for storage
        contradicted_data = []
        
        # Process each example
        for item in tqdm(data, desc=f"Processing {os.path.basename(input_file)}"):
            question = item.get("question", "")
            options = item.get("options", "")
            cot = item.get("cot", "")
            
            # Skip if key components are missing
            if not question or not cot:
                print(f"Skipping incomplete item: {question[:30]}...")
                continue
                
            # Create a copy of the item for contradiction
            contradicted_item = item.copy()
            
            # Inject contradiction
            contradicted_cot = inject_contradiction_with_gpt(question, options, cot)
            contradicted_item["contradicted_cot"] = contradicted_cot
            
            # Add to storage
            contradicted_data.append(contradicted_item)
            
            # Sleep briefly to avoid rate limits if using API
            if os.environ.get("OPENAI_API_KEY"):
                time.sleep(0.2)
                
        # Save results
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(contradicted_data, f, indent=2)
            
        return len(contradicted_data)
            
    except Exception as e:
        print(f"Error processing {input_file}: {e}")
        return 0

def main():
    parser = argparse.ArgumentParser(description="Inject contradictions using OpenAI API")
    parser.add_argument("--model_list", type=str, required=True, help="Path to models.json config file")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing CoT JSON files")
    parser.add_argument("--output_dir", type=str, default="logs/contradicted", help="Directory to save contradicted outputs")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model configurations
    with open(args.model_list, 'r') as f:
        models = json.load(f)
        
    # Process each model's output
    total_examples = 0
    
    for model_id in models:
        model_name = model_id
        
        input_file = os.path.join(args.input_dir, f"{model_name}_{args.dataset}.json")
        output_file = os.path.join(args.output_dir, f"{model_name}_{args.dataset}_contradicted.json")
        
        if os.path.exists(input_file):
            num_examples = process_file(input_file, output_file)
            total_examples += num_examples
            print(f"Processed {model_name}: {num_examples} examples → {output_file}")
        else:
            print(f"Skipping {model_name}: {input_file} not found")
            
    print(f"Processed {total_examples} examples across {len(models)} models")

if __name__ == "__main__":
    main()
