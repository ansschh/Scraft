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
    Optimized to handle larger models with memory constraints.
    
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
    
    # Check model size to apply appropriate optimizations
    is_large_model = any(size in model_id.lower() for size in ["7b", "70b", "65b", "33b", "13b", "llama2"])
    
    # Configure loading parameters
    model_kwargs = {
        "device_map": device_map,
        "torch_dtype": dtype,
        "trust_remote_code": trust_remote_code
    }
    
    # Apply memory optimizations for large models
    if is_large_model:
        print(f"Applying memory optimizations for large model: {model_id}")
        model_kwargs.update({
            "low_cpu_mem_usage": True,
            "offload_folder": "offload_folder",
            "offload_state_dict": True
        })
        
        # Try to clear CUDA cache before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"Cleared CUDA cache. Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load model with optimized parameters
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(model, "gradient_checkpointing_enable") and is_large_model:
            model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled for memory efficiency")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    
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
                params["stop"] = sampling_params.stop
        
        return inputs, params
    
    def generate(self, prompts, sampling_params=None):
        """Generate text for the given prompts using HuggingFace generate() API.
        Memory-optimized for large models.
        
        Args:
            prompts: List of prompt strings
            sampling_params: An optional SamplingParams instance (compatible with vLLM)
            
        Returns:
            List of dictionaries with generated text
        """
        if sampling_params is None:
            sampling_params = SamplingParams()
        
        results = []
        for prompt_idx, prompt in enumerate(prompts):
            # Clear cache between examples for large models
            if torch.cuda.is_available() and "7b" in self.name.lower():
                torch.cuda.empty_cache()
            
            inputs, params = self._prepare_inputs(prompt, sampling_params)
            
            try:
                # Reduce max tokens for larger models to avoid OOM
                max_tokens = params["max_tokens"]
                if "7b" in self.name.lower() and max_tokens > 512:
                    print(f"Reducing max_tokens from {max_tokens} to 512 for 7B model")
                    max_tokens = 512
                
                # Use beam search only for small models (2B and below)
                use_beam_search = params["temperature"] < 0.1 and "2b" in self.name.lower()
                num_beams = 4 if use_beam_search else 1
                
                # Generate with input params mapped to HuggingFace params
                with torch.no_grad():
                    # For 7B, explicitly move inputs to GPU to control memory usage
                    if "7b" in self.name.lower():
                        try:
                            for k, v in inputs.items():
                                if isinstance(v, torch.Tensor):
                                    inputs[k] = v.to("cuda")
                        except Exception as e:
                            print(f"Warning: Issue moving inputs to GPU: {e}")
                    
                    outputs = self.model.generate(
                        **inputs,
                        do_sample=params["temperature"] > 0,
                        temperature=max(params["temperature"], 0.01) if params["temperature"] > 0 else 1.0,
                        top_p=params["top_p"],
                        max_new_tokens=max_tokens,
                        num_beams=num_beams,
                        num_return_sequences=1,
                        pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                        repetition_penalty=1.05,  # Slight penalty for repetition
                        length_penalty=1.0,
                        early_stopping=True,
                        use_cache=True  # Important for memory efficiency
                    )
                
                # Get only the newly generated tokens
                gen_text = self.tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
                
                # Apply stop tokens if needed
                if params["stop"] and len(params["stop"]) > 0:
                    earliest_stop = float("inf")
                    for stop_token in params["stop"]:
                        if stop_token in gen_text:
                            pos = gen_text.find(stop_token)
                            earliest_stop = min(earliest_stop, pos)
                    if earliest_stop < float("inf"):
                        gen_text = gen_text[:earliest_stop]
            
                print(f"Generated text length: {len(gen_text)}")
                results.append({
                    "output": gen_text,  # Use 'output' key for compatibility with existing code
                    "prompt": prompt, 
                    "finished": True,
                    "id": 0  # Placeholder id
                })
            
            except Exception as e:
                print(f"Error generating text for example {prompt_idx}: {e}")
                # Try with even further reduced parameters
                try:
                    print("Retrying with minimal generation parameters...")
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            do_sample=False,  # Greedy decoding
                            max_new_tokens=200,  # Very limited tokens
                            num_beams=1,  # No beam search
                            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                            use_cache=True
                        )
                    
                    gen_text = self.tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
                    print(f"Retry succeeded. Generated length: {len(gen_text)}")
                    
                    results.append({
                        "output": gen_text,  # Use 'output' key for compatibility with existing code
                        "prompt": prompt, 
                        "finished": True,
                        "id": 0  # Placeholder id
                    })
                except Exception as retry_error:
                    print(f"Retry also failed: {retry_error}")
                    # Return empty result on error
                    results.append({
                        "output": "",  # Empty output on error
                        "prompt": prompt,
                        "finished": False,
                        "id": 0  # Placeholder id
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
