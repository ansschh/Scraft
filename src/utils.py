#!/usr/bin/env python3
"""
Common utility functions for SCRaFT benchmarking suite.
Provides model loading, tokenizer configuration, and other shared functionality.
"""

import os
import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set environment variable for CUDA multiprocessing
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def load_model(cfg):
    """
    Load a model from Hugging Face based on configuration.
    Uses Hugging Face Transformers directly without any quantization.
    
    Args:
        cfg: Model configuration dictionary from models.json
        
    Returns:
        HuggingFaceLLMWrapper instance
    """
    model_id = cfg["hf_id"]
    
    print(f"Loading {model_id} using Hugging Face Transformers...")
    
    # Determine precision and device
    dtype = torch.float16
    try:
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
    except:
        pass
    
    trust_remote_code = cfg.get("needs_trust", False)
    device_map = "auto"
    
    # Load with appropriate parameters - NO quantization as requested
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map=device_map,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code
    )
    
    # Create and return the wrapped model
    return HuggingFaceLLMWrapper(model, model_id, safe_tokenizer(cfg))

class HuggingFaceLLMWrapper:
    """
    Wrapper class for Hugging Face models to provide a compatible interface with vLLM.
    This allows for a seamless fallback when vLLM has issues with initialization.
    """
    def __init__(self, model, model_id, tokenizer):
        self.model = model
        self.model_id = model_id
        self.tokenizer = tokenizer
        self.model.eval()  # Set model to evaluation mode
        
        # Set up parameters similar to vLLM defaults
        self.default_params = {
            "temperature": 0.7,
            "top_p": 1.0,
            "max_tokens": 512,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0
        }
    
    def _prepare_inputs(self, prompt, sampling_params=None):
        """Prepare inputs for the model."""
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Extract generation parameters
        params = self.default_params.copy()
        if sampling_params is not None:
            if hasattr(sampling_params, "temperature"):
                params["temperature"] = sampling_params.temperature
            if hasattr(sampling_params, "top_p"):
                params["top_p"] = sampling_params.top_p
            if hasattr(sampling_params, "max_tokens"):
                params["max_tokens"] = sampling_params.max_tokens
            if hasattr(sampling_params, "stop") and sampling_params.stop:
                params["stop_tokens"] = sampling_params.stop
        
        return inputs, params
    
    def generate(self, prompts, sampling_params=None):
        """Generate text completions for the given prompts."""
        results = []
        
        for prompt in prompts:
            inputs, params = self._prepare_inputs(prompt, sampling_params)
            
            try:
                print(f"Generating with params: temp={params['temperature']}, top_p={params['top_p']}, max_tokens={params['max_tokens']}")
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        do_sample=params["temperature"] > 0,
                        temperature=max(params["temperature"], 0.01) if params["temperature"] > 0 else 1.0,  # Avoid 0 temperature issues
                        top_p=params["top_p"],
                        max_new_tokens=params["max_tokens"],
                        num_beams=4 if params["temperature"] < 0.1 else 1,  # Use beam search for low temperatures
                        num_return_sequences=1,
                        pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                        repetition_penalty=1.1,  # Slightly penalize repetition
                        length_penalty=1.0,  # Neither favor nor penalize length
                        early_stopping=True  # Stop when possible
                    )
                
                # Get only the newly generated tokens
                input_length = inputs["input_ids"].shape[1]
                generated_tokens = outputs[0][input_length:]
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                # Apply stop token handling manually
                if "stop_tokens" in params and params["stop_tokens"]:
                    for stop_token in params["stop_tokens"]:
                        if stop_token in generated_text:
                            # Truncate at the stop token
                            generated_text = generated_text.split(stop_token)[0]
                            break
                            
                print(f"Generated text length: {len(generated_text)}")
                
                # Create an output structure similar to vLLM's output
                results.append({
                    "output": generated_text,
                    "prompt": prompt,
                    "finished": True,
                    "id": 0  # Placeholder id
                })
            except Exception as e:
                print(f"Error during generation: {e}")
                # Return empty result on error
                results.append({
                    "output": f"Error: {str(e)}",
                    "prompt": prompt,
                    "finished": False,
                    "id": 0
                })
            
        
        return results


def safe_tokenizer(cfg):
    """
    Load a tokenizer with safe defaults for all model types.
    
    Args:
        cfg: Model configuration dictionary from models.json
        
    Returns:
        Hugging Face tokenizer instance
    """
    model_id = cfg["hf_id"]
    
    # Configure tokenizer options
    tokenizer_kwargs = {
        "use_fast": True,
        "padding_side": "left"
    }
    
    # Handle models that require trust_remote_code
    if cfg.get("needs_trust", False):
        tokenizer_kwargs["trust_remote_code"] = True
    
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
    
    # Ensure pad token is properly set for all model types
    # Some models like Gemma and Phi require this fix
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    # Phi models require right-side padding
    if "phi" in model_id.lower():
        tokenizer.padding_side = "right"
    
    return tokenizer

def format_options(options, model_name=""):
    """
    Format options string based on model preferences.
    Some instruction-tuned models prefer bullet lists.
    
    Args:
        options: Original options string
        model_name: Model name for format customization
        
    Returns:
        Formatted options string
    """
    # Default format is unchanged
    return options
