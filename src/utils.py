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

# Try to import vLLM, but provide fallback if it fails
try:
    from vllm import LLM
    VLLM_AVAILABLE = True
except (ImportError, RuntimeError):
    VLLM_AVAILABLE = False
    warnings.warn("vLLM not available or failed to initialize. Using Hugging Face Transformers as fallback.")

def load_model(cfg):
    """
    Load a model from Hugging Face based on configuration.
    Handles tensor parallelism, trust_remote_code, and quantization.
    Can use vLLM if available or fall back to Hugging Face Transformers.
    
    Args:
        cfg: Model configuration dictionary from models.json
        
    Returns:
        Either a vLLM LLM instance or a HuggingFaceLLMWrapper
    """
    model_id = cfg["hf_id"]
    
    # Try to use vLLM first if available
    if VLLM_AVAILABLE:
        try:
            tp_size = cfg.get("tp", 1)  # Default to 1 if not specified
            
            # Configure model options
            model_kwargs = {
                "model": model_id,
                "tensor_parallel_size": tp_size
            }
            
            # Configure dtype - vLLM versions before 0.3.0 don't support 4-bit quantization
            # so we just use float16 or bfloat16 based on CUDA capabilities
            try:
                if torch.cuda.is_bf16_supported():
                    model_kwargs["dtype"] = "bfloat16"
                else:
                    model_kwargs["dtype"] = "float16"
            except:
                # Fallback to float16 if bf16 check fails
                model_kwargs["dtype"] = "float16"
            
            # Handle models that require trust_remote_code
            if cfg.get("needs_trust", False):
                model_kwargs["trust_remote_code"] = True
            
            # Handle Mixtral-specific optimizations
            if "mixtral" in model_id.lower():
                model_kwargs["gpu_memory_utilization"] = 0.85
            
            # Create and return the LLM instance
            return LLM(**model_kwargs)
        except Exception as e:
            print(f"vLLM initialization failed with error: {e}")
            print("Falling back to HuggingFace Transformers...")
    
    # Fallback to Hugging Face Transformers
    print(f"Loading {model_id} using Hugging Face Transformers as fallback...")
    
    # Determine precision and device
    dtype = torch.float16
    try:
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
    except:
        pass
    
    trust_remote_code = cfg.get("needs_trust", False)
    device_map = "auto"
    
    # Try 8-bit quantization if model is large
    use_8bit = False
    if any(size in model_id.lower() for size in ["70b", "65b", "33b", "34b", "40b"]):
        use_8bit = True
        
    # Load with appropriate parameters
    if use_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            load_in_8bit=True,
            trust_remote_code=trust_remote_code
        )
    else:
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
        
        return inputs, params
    
    def generate(self, prompts, sampling_params=None):
        """Generate text completions for the given prompts."""
        results = []
        
        for prompt in prompts:
            inputs, params = self._prepare_inputs(prompt, sampling_params)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    do_sample=params["temperature"] > 0,
                    temperature=params["temperature"],
                    top_p=params["top_p"],
                    max_new_tokens=params["max_tokens"],
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
                )
            
            # Get only the newly generated tokens
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Create an output structure similar to vLLM's output
            results.append({
                "output": generated_text,
                "prompt": prompt,
                "finished": True,
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
