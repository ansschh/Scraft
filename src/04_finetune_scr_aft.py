#!/usr/bin/env python3
"""
Runs Direct Preference Optimisation with LoRA adapters.
Uses 4-bit quantization for base weights with gradients flowing only through LoRA layers.
"""

import os
import torch
import argparse
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer

def main():
    parser = argparse.ArgumentParser(description="Fine-tune with SCRaFT using DPO and LoRA")
    parser.add_argument("--model_list", type=str, required=True, help="Path to configs/models.json")
    parser.add_argument("--model_size", type=str, default="8B", help="Model size to fine-tune (e.g., 8B)")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the DPO dataset")
    parser.add_argument("--output_dir", type=str, default="ft-scraft", help="Directory to save fine-tuned model")
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA r-rank")
    parser.add_argument("--lora_alpha", type=int, default=128, help="LoRA alpha scaling factor")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--beta", type=float, default=1.0, help="DPO beta parameter")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max training steps (-1 for full epochs)")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--min_dataset_size", type=int, default=100, help="Minimum dataset size to fine-tune with")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load DPO dataset
    train_dataset = load_from_disk(args.dataset_path)
    print(f"Loaded dataset with {len(train_dataset)} examples")
    
    # Skip fine-tuning if dataset is too small
    if len(train_dataset) < args.min_dataset_size:
        print(f"Dataset size ({len(train_dataset)}) is smaller than --min_dataset_size ({args.min_dataset_size}). Skipping fine-tuning.")
        return
    
    # Load model configurations
    from utils import safe_tokenizer, load_model
    import json as j
    model_cfgs = j.load(open(args.model_list))
    
    # Find target model based on size (we only fine-tune one model size for SCRaFT)
    target_model_cfg = None
    target_model_name = None
    
    for name, cfg in model_cfgs.items():
        if args.model_size in name:
            target_model_cfg = cfg
            target_model_name = name
            break
    
    if not target_model_cfg:
        print(f"No model with size {args.model_size} found in model_list. Exiting.")
        return
    
    model_id = target_model_cfg["id"]
    print(f"Fine-tuning model: {target_model_name} ({model_id})")
    
    # Final output paths
    model_output_dir = os.path.join(args.output_dir, f"scraft-{args.model_size}")
    adapter_output_dir = os.path.join(model_output_dir, "adapter")
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Load tokenizer using safe_tokenizer helper function
    tokenizer = safe_tokenizer(target_model_cfg)
    
    # Configure LoRA
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Sweet spot for affecting token-level generation
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Load model with 4-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        load_in_4bit=True,  # Quantized base weights
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=target_model_cfg.get("trust_remote_code", False),
    )
    model = get_peft_model(model, peft_config)
    model.config.use_cache = False  # Disable KV cache during training
    
    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        optim="paged_adamw_32bit",
        max_steps=args.max_steps,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",  # Default to no logging for batch runs
        remove_unused_columns=False,
    )
    
    # Configure DPO trainer
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Use original weights as reference
        args=training_args,
        beta=args.beta,  # Controls KL penalty; β=1.0 balances preference and KL
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        max_length=512,  # Maximum sequence length
        max_prompt_length=256,  # Maximum length for input prompts
        max_target_length=256,  # Maximum length for output responses
    )
    
    # Run training
    dpo_trainer.train()
    
    # Save final model and tokenizer
    dpo_trainer.save_model(model_output_dir)
    print(f"Model saved to {model_output_dir}")
    
    # Save adapter configuration separately for easier loading
    model.save_pretrained(adapter_output_dir)
    print(f"LoRA adapter saved to {adapter_output_dir}")

if __name__ == "__main__":
    main()
