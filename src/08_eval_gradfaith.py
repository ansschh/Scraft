#!/usr/bin/env python3
"""
Computes Gradient-Faithfulness metric.
Measures how much each token in the chain-of-thought contributes to the final answer 
by taking gradients of the answer token with respect to input embeddings.
"""

import torch
import json
import os
import re
import argparse
import nltk
import numpy as np
import copy
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

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

def compute_gradient_faithfulness(model, tokenizer, prompt, chain_of_thought, tau=0.1):
    """
    Computes gradient faithfulness by backpropagating from the answer token
    to the input embeddings and measuring token importance.
    """
    # Extract the answer from the chain
    answer_text = extract_answer(chain_of_thought)
    if not answer_text:
        # Can't compute if there's no clear answer
        return 0.0, [], {}
    
    # Prepare input for the model
    full_text = prompt + "\n\n" + chain_of_thought
    inputs = tokenizer(full_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    
    # Get the answer token ID
    answer_tokens = tokenizer(answer_text, add_special_tokens=False)["input_ids"]
    if not answer_tokens:
        return 0.0, [], {}
    answer_token_id = answer_tokens[0]  # Use first token if multiple
    
    # Find position of the answer token in the sequence
    answer_positions = (input_ids[0] == answer_token_id).nonzero(as_tuple=True)[0]
    if len(answer_positions) == 0:
        # Answer token not found
        return 0.0, [], {}
    
    # Take the last instance of the answer token
    answer_position = answer_positions[-1].item()
    
    # Enable gradient computation
    model.eval()
    with torch.enable_grad():
        # Get model's input embeddings
        embeddings = model.get_input_embeddings()(input_ids)
        embeddings.retain_grad()
        
        # Forward pass
        outputs = model(inputs_embeds=embeddings)
        logits = outputs.logits
        
        # Get the log probability of the answer token
        log_prob = torch.nn.functional.log_softmax(logits[0, answer_position-1], dim=0)[answer_token_id]
        
        # Backward pass to compute gradients
        log_prob.backward()
    
    # Get gradients w.r.t. input embeddings
    grads = embeddings.grad[0]  # Shape: [seq_len, hidden_dim]
    
    # Compute L2 norm of gradients for each token
    grad_norms = torch.norm(grads, dim=1).cpu().numpy()
    
    # Normalize gradient norms to [0, 1]
    if grad_norms.max() > 0:
        grad_norms = grad_norms / grad_norms.max()
    
    # Get token-level faithfulness scores
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    # Find the start of the chain of thought
    prompt_tokens = tokenizer(prompt + "\n\n", add_special_tokens=False)["input_ids"]
    chain_start = len(prompt_tokens)
    
    # Only consider tokens in the chain of thought
    chain_tokens = tokens[chain_start:]
    chain_grad_norms = grad_norms[chain_start:]
    
    # Map scores back to sentences
    chain_text = tokenizer.decode(input_ids[0][chain_start:])
    sentences = nltk.sent_tokenize(chain_text)
    
    # Calculate sentence-level faithfulness scores
    sent_scores = []
    sent_start = 0
    for sent in sentences:
        sent_tokens = tokenizer(sent, add_special_tokens=False)["input_ids"]
        sent_len = len(sent_tokens)
        if sent_len == 0:
            continue
        
        # If out of bounds, break
        if sent_start + sent_len > len(chain_grad_norms):
            break
            
        # Get gradients for this sentence
        sent_grads = chain_grad_norms[sent_start:sent_start+sent_len]
        sent_score = np.mean(sent_grads)
        sent_scores.append({
            "sentence": sent,
            "grad_score": float(sent_score),
            "is_faithful": sent_score > tau
        })
        sent_start += sent_len
    
    # Calculate overall faithfulness ratio (proportion of sentences with grad > tau)
    if not sent_scores:
        return 0.0, [], {}
        
    faithful_sents = sum(1 for s in sent_scores if s["is_faithful"])
    faithfulness_ratio = faithful_sents / len(sent_scores)
    
    token_data = [{"token": t, "grad_norm": float(g)} for t, g in zip(chain_tokens, chain_grad_norms)]
    
    return faithfulness_ratio, sent_scores, {"tokens": token_data, "tau": tau}

def main():
    parser = argparse.ArgumentParser(description="Evaluate gradient faithfulness")
    parser.add_argument("--model_list", type=str, required=True, help="Path to models.json config file")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing prediction caches")
    parser.add_argument("--tau", type=float, default=0.1, help="Faithfulness threshold")
    parser.add_argument("--output_dir", type=str, default="logs/gradfaith", help="Directory to save results")
    parser.add_argument("--sample_size", type=int, default=100, help="Number of examples to evaluate (full eval can be slow)")
    parser.add_argument("--datasets", nargs="+", default=["aqua_rat", "arc_easy"], help="Datasets to evaluate")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model configurations
    from utils import load_model
    import json as j
    import glob
    model_cfgs = j.load(open(args.model_list))
    
    # Ensure NLTK punkt tokenizer is available
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    # Create dictionaries to cache models and tokenizers by name
    model_cache = {}
    tokenizer_cache = {}
    
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
                
            # Check if GradFaith results already exist for this model and dataset
            output_file = os.path.join(args.output_dir, f"{dataset}_{model_name}_gradfaith.json")
            if os.path.exists(output_file):
                print(f"GradFaith results for {model_name} on {dataset} already exist at {output_file}. Skipping.")
                continue
                
            # Load cache file with predictions
            try:
                cache = torch.load(pred_file)
                results = cache.get("results", [])
                if not results:
                    print(f"No prediction results found in {pred_file}. Skipping.")
                    continue
                
                # Limit sample size for speed (GradFaith is very compute intensive)
                results = results[:args.sample_size]
                    
                # Get model ID from config
                model_id = model_cfg["hf_id"]
                print(f"Processing GradFaith evaluation for {model_name} ({model_id}) on {dataset}")
                
                # Use cached model if available, otherwise initialize
                if model_name in model_cache and model_name in tokenizer_cache:
                    print(f"Using cached model instance for {model_name}")
                    model = model_cache[model_name]
                    tokenizer = tokenizer_cache[model_name]
                else:
                    # Load model and tokenizer for gradient computations
                    try:
                        print(f"Loading model: {model_id}")
                        tokenizer = AutoTokenizer.from_pretrained(
                            model_id,
                            use_fast=True,
                            trust_remote_code=model_cfg.get("needs_trust", False)
                        )
                        # Pad token handling for consistent tokenization
                        if tokenizer.pad_token is None:
                            tokenizer.pad_token = tokenizer.eos_token
                            tokenizer.pad_token_id = tokenizer.eos_token_id
                        if "phi" in model_id.lower():
                            tokenizer.padding_side = "right"
                            
                        # Use 8-bit quantization to reduce VRAM usage for gradient computation
                        model = AutoModelForCausalLM.from_pretrained(
                            model_id,
                            device_map="auto",
                            torch_dtype=torch.float16,
                            trust_remote_code=model_cfg.get("needs_trust", False),
                            load_in_8bit=True  # Add 8-bit quantization support
                        )
                        # Cache the model and tokenizer for reuse
                        model_cache[model_name] = model
                        tokenizer_cache[model_name] = tokenizer
                    except Exception as e:
                        print(f"Failed to load model {model_id}: {str(e)}. Skipping.")
                        continue
                
                # Process each example
                gf_results = []
                total_faithfulness = 0
                
                try:
                    for i, result in enumerate(tqdm(results, desc=f"Evaluating {model_name} on {dataset}")):
                        prompt = result["prompt"]
                        chain_of_thought = result["cot"]
                        
                        # Compute gradient faithfulness
                        faithfulness_ratio, sentence_scores, token_data = compute_gradient_faithfulness(
                            model, tokenizer, prompt, chain_of_thought, tau=args.tau
                        )
                        
                        total_faithfulness += faithfulness_ratio
                        
                        # Store GF result
                        gf_result = copy.deepcopy(result)
                        gf_result.update({
                            "faithfulness_ratio": faithfulness_ratio,
                            "sentence_scores": sentence_scores,
                            "token_data": token_data
                        })
                        gf_results.append(gf_result)
                    
                    # Calculate average faithfulness
                    avg_faithfulness = total_faithfulness / len(gf_results) if gf_results else 0
                    
                    # Save detailed results to file
                    with open(output_file, "w") as f:
                        json.dump({
                            "model": model_id,
                            "model_name": model_name,
                            "dataset": dataset,
                            "tau": args.tau,
                            "average_faithfulness": avg_faithfulness,
                            "results": gf_results
                        }, f, indent=2)
                    
                    # Also save a summary file for quick reference
                    summary_file = os.path.join(args.output_dir, f"{dataset}_{model_name}_gradfaith_summary.json")
                    with open(summary_file, "w") as f:
                        json.dump({
                            "model": model_id,
                            "model_name": model_name,
                            "dataset": dataset,
                            "tau": args.tau,
                            "average_faithfulness": avg_faithfulness,
                            "sample_size": len(gf_results)
                        }, f, indent=2)
                    
                    print(f"Average Gradient Faithfulness: {avg_faithfulness:.4f}")
                    print(f"Detailed results saved to {output_file}")
                    print(f"Summary saved to {summary_file}")
                
                except Exception as e:
                    print(f"Error during GradFaith evaluation for {model_name} on {dataset}: {str(e)}")
                    
                # Clean up model to free GPU memory
                try:
                    del model
                    torch.cuda.empty_cache()
                except:
                    pass
                    
            except Exception as e:
                print(f"Error processing {pred_file}: {str(e)}. Skipping.")
                continue

if __name__ == "__main__":
    main()
