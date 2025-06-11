#!/usr/bin/env python3
"""
Improved answer extraction using OpenAI API for the SCRaFT pipeline.
This script extracts final answers from chain-of-thought reasoning using GPT API,
which is more reliable than regex patterns for complex reasoning chains.
"""

import os
import json
import argparse
from tqdm import tqdm
import re
import time
import openai
from openai import OpenAI
from pathlib import Path
import glob

# Setup OpenAI client
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY", ""),
)

def extract_answer_with_gpt(cot_text, question, options):
    """
    Extract the final answer from chain-of-thought reasoning using GPT.
    Much more reliable than regex for complex reasoning chains.
    """
    # If API key not set, fall back to regex method
    if not os.environ.get("OPENAI_API_KEY"):
        return extract_answer_regex(cot_text)
    
    system_message = """
    You are an AI assistant tasked with extracting the final answer from a chain-of-thought reasoning.
    Extract ONLY the letter (A, B, C, D, E) or number (single digit) that represents the final answer.
    Return ONLY the letter or number, with no explanation or additional text.
    If multiple answers are given, extract only the final one.
    If no clear answer can be determined, respond with "NONE".
    """
    
    user_message = f"""
    QUESTION: {question}
    
    OPTIONS: {options}
    
    REASONING CHAIN:
    {cot_text}
    
    Extract ONLY the letter or number that represents the final answer:
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", # Fast and cost-effective
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.0,  # Deterministic
            max_tokens=5,     # Very short answer
        )
        
        answer = response.choices[0].message.content.strip().upper()
        # If answer is not a valid option (single letter A-E or digit), return empty
        if not re.match(r'^([A-E]|\d)$', answer):
            if answer == "NONE":
                return ""
            else:
                print(f"Invalid answer format: {answer}")
                return extract_answer_regex(cot_text)  # Fall back to regex
                
        return answer
        
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return extract_answer_regex(cot_text)  # Fall back to regex

def extract_answer_regex(cot_text):
    """
    Fallback regex-based answer extraction if API call fails.
    """
    # Look for "Answer: X" or "(X)" pattern at the end
    answer_pattern = r'Answer:\s*([A-E]|\d+)'
    alt_pattern = r'\(([A-E]|\d+)\)'
    another_pattern = r'[Tt]he\s+answer\s+is\s+([A-E]|\d+)'
    final_pattern = r'[Ff]inal\s+answer\s*:?\s*([A-E]|\d+)'
    
    for pattern in [answer_pattern, alt_pattern, another_pattern, final_pattern]:
        match = re.search(pattern, cot_text)
        if match:
            return match.group(1)
            
    # Check for last option mentioned
    option_pattern = r'[Oo]ption\s+([A-E]|\d+)'
    matches = list(re.finditer(option_pattern, cot_text))
    if matches:
        return matches[-1].group(1)
    
    # Look for single letter or digit at end of text
    last_sentences = cot_text.strip().split('.')[-2:]
    for sentence in last_sentences:
        match = re.search(r'\b([A-E]|\d+)\b', sentence)
        if match:
            return match.group(1)
    
    return ""

def process_file(input_file, output_file):
    """
    Process a single CoT file, extract answers, and save to output file.
    """
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
            
        # Extract answers for each example
        for item in tqdm(data, desc=f"Processing {os.path.basename(input_file)}"):
            question = item.get("question", "")
            options = item.get("options", "")
            cot = item.get("cot", "")
            
            # Extract answer using GPT API
            answer = extract_answer_with_gpt(cot, question, options)
            item["extracted_answer"] = answer
            
        # Save results
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
            
        return len(data)
            
    except Exception as e:
        print(f"Error processing {input_file}: {e}")
        return 0

def main():
    parser = argparse.ArgumentParser(description="Extract answers from CoT using OpenAI API")
    parser.add_argument("--model_list", type=str, required=True, help="Path to models.json config file")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing CoT JSON files")
    parser.add_argument("--output_dir", type=str, default="logs/answers", help="Directory to save extracted answers")
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
        output_file = os.path.join(args.output_dir, f"{model_name}_{args.dataset}_answers.json")
        
        if os.path.exists(input_file):
            num_examples = process_file(input_file, output_file)
            total_examples += num_examples
            print(f"Processed {model_name}: {num_examples} examples → {output_file}")
        else:
            print(f"Skipping {model_name}: {input_file} not found")
            
    print(f"Processed {total_examples} examples across {len(models)} models")

if __name__ == "__main__":
    main()
