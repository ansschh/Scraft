#!/usr/bin/env python3
"""
SCRaFT Multi-Model Benchmarking Orchestrator
Runs the entire SCRaFT pipeline for multiple models and datasets
"""

import os
import argparse
import subprocess
import json
from pathlib import Path

def ensure_dir(path):
    """Create directory if it doesn't exist"""
    Path(path).mkdir(parents=True, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description="Run SCRaFT multi-model benchmarking")
    parser.add_argument("--model_list", type=str, default="configs/models.json", 
                        help="Path to models.json config file")
    parser.add_argument("--datasets", type=str, nargs="+", 
                        default=["aqua_rat", "arc_easy"], 
                        help="Datasets to benchmark")
    parser.add_argument("--model_sizes", type=str, nargs="+", 
                        default=["8B"], 
                        help="Model size categories to evaluate")
    parser.add_argument("--steps", type=str, nargs="+", 
                        default=["generate", "contradict", "refine", "pairs", "finetune", "accuracy", "cst", "sns", "gradfaith", "plot"],
                        help="Steps to run: generate, contradict, refine, pairs, finetune, accuracy, cst, sns, gradfaith, plot")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for generation and inference")
    parser.add_argument("--output_dir", type=str, default="logs", help="Base directory for outputs")
    parser.add_argument("--dryrun", action="store_true", help="Print commands without running")
    args = parser.parse_args()
    
    # Ensure directories exist
    for dataset in args.datasets:
        ensure_dir(os.path.join(args.output_dir, dataset))
    ensure_dir(os.path.join(args.output_dir, "cot"))
    ensure_dir(os.path.join(args.output_dir, "cst"))
    ensure_dir(os.path.join(args.output_dir, "sns"))
    ensure_dir(os.path.join(args.output_dir, "gradfaith"))
    ensure_dir(os.path.join(args.output_dir, "plots"))
    
    # Load model configurations
    with open(args.model_list) as f:
        models = json.load(f)
    
    # Process model sizes (normalize to uppercase)
    model_sizes = [size.upper() for size in args.model_sizes]
    
    print(f"🚀 Running SCRaFT benchmarking pipeline for {len(models)} models on {len(args.datasets)} datasets")
    print(f"🔧 Selected steps: {args.steps}")
    
    # Step 1: Generate Chain-of-Thought (CoT)
    if "generate" in args.steps:
        print("\n📝 Step 1: Generating Chain-of-Thought (CoT) responses...")
        for model_name, config in models.items():
            # Check if model matches requested sizes
            model_size_str = f"{config['params']}B"
            matches_size = any(size in model_size_str for size in model_sizes)
            if not matches_size and model_sizes != ["ALL"]:
                continue
                
            for dataset in args.datasets:
                cmd = [
                    "python", "src/00_generate_cot.py",
                    "--model", config["hf_id"],
                    "--dataset", dataset,
                    "--batch_size", str(args.batch_size),
                    "--output_dir", os.path.join(args.output_dir, "cot")
                ]
                if config.get("needs_trust", False):
                    cmd.append("--trust_remote_code")
                if config.get("tp", 1) > 1:
                    cmd.extend(["--tensor_parallel", str(config["tp"])])
                
                print(f"Running: {' '.join(cmd)}")
                if not args.dryrun:
                    subprocess.run(cmd)
    
    # Step 2: Inject Contradictions
    if "contradict" in args.steps:
        print("\n🔄 Step 2: Injecting contradictions...")
        for dataset in args.datasets:
            for model_name in models:
                # Check if model matches requested sizes
                model_size_str = f"{models[model_name]['params']}B"  
                matches_size = any(size in model_size_str for size in model_sizes)
                if not matches_size and model_sizes != ["ALL"]:
                    continue
                    
                cmd = [
                    "python", "src/01_inject_contradiction.py",
                    "--cache", os.path.join(args.output_dir, "cot", f"{dataset}_{model_name}_predictions.pkl"),
                    "--output_dir", args.output_dir
                ]
                print(f"Running: {' '.join(cmd)}")
                if not args.dryrun:
                    subprocess.run(cmd)
    
    # Step 3: Self-Refine
    if "refine" in args.steps:
        print("\n🔍 Step 3: Self-refinement...")
        for model_name, config in models.items():
            # We only use 8B models as refiners in the original SCRaFT design
            model_size = config.get("params", 0)
            if not any(str(model_size) in size for size in model_sizes) and model_sizes != ["ALL"]:
                continue
                
            for dataset in args.datasets:
                cmd = [
                    "python", "src/02_self_refine.py",
                    "--model", config["hf_id"],
                    "--dataset", dataset,
                    "--batch_size", str(args.batch_size),
                    "--output_dir", args.output_dir
                ]
                if config.get("needs_trust", False):
                    cmd.append("--trust_remote_code")
                if config.get("tp", 1) > 1:
                    cmd.extend(["--tensor_parallel", str(config["tp"])])
                
                print(f"Running: {' '.join(cmd)}")
                if not args.dryrun:
                    subprocess.run(cmd)
    
    # Step 4: Build Pairs
    if "pairs" in args.steps:
        print("\n👥 Step 4: Building training pairs...")
        for dataset in args.datasets:
            # Match refiner model to dataset, based on available caches from step 3
            for model_name in models:
                # Check if model matches requested sizes
                model_size_str = f"{models[model_name]['params']}B"
                matches_size = any(size in model_size_str for size in model_sizes)
                if not matches_size and model_sizes != ["ALL"]:
                    continue
                    
                cmd = [
                    "python", "src/03_build_pairs.py",
                    "--refiner", model_name,
                    "--dataset", dataset,
                    "--output_dir", args.output_dir
                ]
                print(f"Running: {' '.join(cmd)}")
                if not args.dryrun:
                    subprocess.run(cmd)
    
    # Step 5: Fine-tune
    if "finetune" in args.steps:
        print("\n🧠 Step 5: Fine-tuning with LoRA...")
        cmd = [
            "python", "src/04_finetune_scr_aft.py",
            "--model_list", args.model_list,
            "--model_size", args.model_sizes[0] if len(args.model_sizes) == 1 else "8B",
            "--datasets", *args.datasets,
            "--output_dir", args.output_dir,
            "--batch_size", str(args.batch_size)
        ]
        print(f"Running: {' '.join(cmd)}")
        if not args.dryrun:
            subprocess.run(cmd)
    
    # Step 6: Evaluate Accuracy
    if "accuracy" in args.steps:
        print("\n📊 Step 6: Evaluating accuracy...")
        cmd = [
            "python", "src/05_eval_accuracy.py",
            "--model_list", args.model_list,
            "--input_dir", args.output_dir,
            "--datasets", *args.datasets,
            "--output_dir", os.path.join(args.output_dir, "accuracy")
        ]
        print(f"Running: {' '.join(cmd)}")
        if not args.dryrun:
            subprocess.run(cmd)
    
    # Step 7: Evaluate CST (Contradiction Sensitivity Test)
    if "cst" in args.steps:
        print("\n🧪 Step 7: Evaluating CST (Contradiction Sensitivity Test)...")
        cmd = [
            "python", "src/06_eval_cst.py",
            "--model_list", args.model_list,
            "--input_dir", args.output_dir,
            "--datasets", *args.datasets,
            "--output_dir", os.path.join(args.output_dir, "cst")
        ]
        print(f"Running: {' '.join(cmd)}")
        if not args.dryrun:
            subprocess.run(cmd)
    
    # Step 8: Evaluate SNS (Sentence Necessity Search)
    if "sns" in args.steps:
        print("\n🔍 Step 8: Evaluating SNS (Sentence Necessity Search)...")
        cmd = [
            "python", "src/07_eval_sns.py",
            "--model_list", args.model_list,
            "--input_dir", args.output_dir,
            "--datasets", *args.datasets,
            "--output_dir", os.path.join(args.output_dir, "sns")
        ]
        print(f"Running: {' '.join(cmd)}")
        if not args.dryrun:
            subprocess.run(cmd)
    
    # Step 9: Evaluate GradFaith (Gradient Faithfulness)
    if "gradfaith" in args.steps:
        print("\n📈 Step 9: Evaluating GradFaith (Gradient Faithfulness)...")
        cmd = [
            "python", "src/08_eval_gradfaith.py",
            "--model_list", args.model_list,
            "--input_dir", args.output_dir,
            "--datasets", *args.datasets,
            "--output_dir", os.path.join(args.output_dir, "gradfaith"),
            "--sample_size", "50"  # GradFaith is compute-intensive, so use a smaller sample
        ]
        print(f"Running: {' '.join(cmd)}")
        if not args.dryrun:
            subprocess.run(cmd)
    
    # Step 10: Plot Scaling Results
    if "plot" in args.steps:
        print("\n📊 Step 10: Plotting scaling results...")
        cmd = [
            "python", "src/09_plot_scaling.py",
            "--model_list", args.model_list,
            "--input_dir", args.output_dir,
            "--output_dir", os.path.join(args.output_dir, "plots")
        ]
        print(f"Running: {' '.join(cmd)}")
        if not args.dryrun:
            subprocess.run(cmd)
    
    print("\n✅ SCRaFT benchmarking pipeline completed!")

if __name__ == "__main__":
    main()
