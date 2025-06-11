#!/usr/bin/env python3
"""
Common utility functions for SCRaFT benchmarking suite.
Provides model loading, tokenizer configuration, and other shared functionality.
"""

import torch
from vllm import LLM
from transformers import AutoTokenizer

def load_model(cfg):
    """
    Load a model from Hugging Face based on configuration.
    Handles tensor parallelism, trust_remote_code, and quantization.
    
    Args:
        cfg: Model configuration dictionary from models.json
        
    Returns:
        vLLM LLM instance
    """
    model_id = cfg["hf_id"]
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
