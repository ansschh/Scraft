#!/usr/bin/env python3
"""Common utility functions for SCRaFT benchmarking suite.
Provides model loading, tokenizer configuration, and other shared functionality.
Optimized for small models and fast execution.
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
    params_size = cfg.get("params", 0)
    
    # Only models >5B are considered large models
    is_large_model = params_size > 5.0 or any(size in model_id.lower() for size in ["7b", "70b", "65b", "33b", "13b", "llama2"])
    
    # For small models (≤3B params), use simple loading for speed
    if not is_large_model and params_size <= 3.0:
        print(f"Fast loading for small model: {model_id} ({params_size}B params)")
        model_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": trust_remote_code
        }
        # For truly tiny models, we can use the legacy loading path
        if params_size <= 1.5:
            print(f"Using single-device loading for tiny model: {model_id}")
            device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
            model_kwargs["device_map"] = device_map
        else:
            model_kwargs["device_map"] = "auto"
    else:
        print(f"Using memory-optimized loading for larger model: {model_id}")
        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": dtype,
            "trust_remote_code": trust_remote_code,
            "low_cpu_mem_usage": True
        }
        
        # Additional optimizations only for truly large models (>5B params)
        if is_large_model:
            print(f"Applying heavy memory optimizations for large model: {model_id}")
            model_kwargs.update({
                "offload_folder": "offload_folder",
                "offload_state_dict": True
            })
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
        self.name = model_id  # Add name attribute for compatibility with our code
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
    
    def generate(self, prompts, sampling_params, prompt_idx=0):
        """Generate outputs for the given prompts."""
        results = []
        
        params = {
            "temperature": sampling_params.temperature,
            "top_p": sampling_params.top_p,
            "max_tokens": sampling_params.max_tokens,
            "stop": sampling_params.stop
        }
        
        # Get model param size
        model_size = 0
        if hasattr(self, 'model_id'):
            if 'gemma-2b' in self.model_id or '2b' in self.model_id:
                model_size = 2
            elif '1.3b' in self.model_id or '1b' in self.model_id or '1.5' in self.model_id:
                model_size = 1
            elif '7b' in self.model_id or '13b' in self.model_id:
                model_size = 7
        
        # Process each prompt
        for prompt_idx, prompt in enumerate(prompts):
            # Only clear cache for larger models
            if model_size >= 3 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Adapt generation parameters based on model size
            max_tokens = params["max_tokens"]
            if model_size <= 3:
                # Smaller models can generate efficiently with default settings
                pass
            elif model_size >= 7:
                # For larger models, reduce the max tokens
                max_tokens = min(512, max_tokens)
                print(f"Reduced max tokens to {max_tokens} for large model")
            
            # Optimize beam search for small models
            num_beams = 1
            if model_size <= 3 and params["temperature"] < 0.3:
                # Use efficient beam search for small models with moderate temperature
                num_beams = 3
            
            # Tokenize input with padding to make compatible across all models
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            
            # Generate text with error handling
            try:
                # For 7B, explicitly move inputs to GPU to control memory usage
                if "7b" in self.name.lower():
                    try:
                        for k, v in inputs.items():
                            if isinstance(v, torch.Tensor):
                                inputs[k] = v.to("cuda")
                    except Exception as e:
                        print(f"Warning: Issue moving inputs to GPU: {e}")
                
                # Use beam search only for small models (2B and below)
                use_beam_search = params["temperature"] < 0.1 and "2b" in self.name.lower()
                
                # Generate with input params mapped to HuggingFace params
                with torch.no_grad():
                    # Remove early_stopping from generation flags for models that don't support it
                    gen_kwargs = {
                        "do_sample": params["temperature"] > 0,
                        "temperature": max(params["temperature"], 0.01) if params["temperature"] > 0 else 1.0,
                        "top_p": params["top_p"],
                        "max_new_tokens": max_tokens,
                        "num_beams": num_beams,
                    "num_return_sequences": 1,
                    "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    "repetition_penalty": 1.05,  # Slight penalty for repetition
                    "length_penalty": 1.0,
                    "use_cache": True  # Important for memory efficiency
                }
                    
                    # Only add early_stopping for models that support it (not OPT)
                if not "opt" in self.name.lower():
                        gen_kwargs["early_stopping"] = True
                    
                outputs = self.model.generate(**inputs, **gen_kwargs)
                
                # Get only the newly generated tokens - handling both tensor and list output formats
                try:
                    if isinstance(outputs, torch.Tensor):
                        input_length = inputs["input_ids"].shape[1] if isinstance(inputs["input_ids"], torch.Tensor) else len(inputs["input_ids"][0])
                        gen_text = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
                    else:
                        gen_text = self.tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
                except IndexError:
                    # Fallback for different output formats
                    gen_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    # Remove the input prompt from the beginning
                    original_text = self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
                    if gen_text.startswith(original_text):
                        gen_text = gen_text[len(original_text):]
                
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
                    
                    # Handle output formats consistently using the same approach as in main generation
                    try:
                        if isinstance(outputs, torch.Tensor):
                            input_length = inputs["input_ids"].shape[1] if isinstance(inputs["input_ids"], torch.Tensor) else len(inputs["input_ids"][0])
                            gen_text = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
                        else:
                            gen_text = self.tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
                    except IndexError:
                        # Fallback for different output formats
                        gen_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                        # Remove the input prompt from the beginning
                        original_text = self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
                        if gen_text.startswith(original_text):
                            gen_text = gen_text[len(original_text):]
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
