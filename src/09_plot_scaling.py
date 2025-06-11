#!/usr/bin/env python3
"""
Plotting script for SCRaFT benchmarking suite.
Generates:
1. CST vs. Model Size curve (log-scale)
2. Accuracy vs. CST Pareto frontier plot
3. CSV summary for paper tables
"""

import json
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def load_json(filepath):
    """Load a JSON file and return its contents."""
    with open(filepath, 'r') as f:
        return json.load(f)

def collect_results(logs_dir, model_config):
    """
    Collect all evaluation results and merge them into a DataFrame.
    
    Args:
        logs_dir: Directory containing evaluation logs
        model_config: Path to models.json config file
        
    Returns:
        DataFrame with merged evaluation metrics
    """
    # Load model configurations
    models = load_json(model_config)
    
    # Initialize data storage
    data = []
    
    # Find all summary files
    logs_path = Path(logs_dir)
    
    # Look in subdirectories for evaluation results
    cst_dir = logs_path / "cst"
    sns_dir = logs_path / "sns"
    acc_dir = logs_path / "accuracy"
    gradfaith_dir = logs_path / "gradfaith"
    
    # If the directories don't exist, check the main logs directory
    if not cst_dir.exists():
        cst_files = list(logs_path.glob("*_cst_summary.json"))
    else:
        cst_files = list(cst_dir.glob("*_cst_summary.json"))
        
    if not sns_dir.exists():
        sns_files = list(logs_path.glob("*_sns_summary.json"))
    else:
        sns_files = list(sns_dir.glob("*_sns_summary.json"))
        
    if not acc_dir.exists():
        acc_files = list(logs_path.glob("*_accuracy.json"))
    else:
        acc_files = list(acc_dir.glob("*_accuracy.json"))
        
    if not gradfaith_dir.exists():
        gradfaith_files = list(logs_path.glob("*_gradfaith_summary.json"))
    else:
        gradfaith_files = list(gradfaith_dir.glob("*_gradfaith_summary.json"))
    
    # Process CST results
    for cst_file in cst_files:
        cst_data = load_json(cst_file)
        model_name = cst_data.get("model_name", cst_data["model"])
        
        # Extract model base name from path if it's a local model
        if os.path.sep in model_name:
            model_name = os.path.basename(model_name)
        
        # Get dataset name
        dataset = cst_data["dataset"]
        
        # Find corresponding accuracy file
        acc_file = next((f for f in acc_files if dataset in f.name and model_name in f.name), None)
        if acc_file:
            acc_data = load_json(acc_file)
            accuracy = acc_data["accuracy"]
        else:
            accuracy = None
        
        # Find corresponding SNS file
        sns_file = next((f for f in sns_files if dataset in f.name and model_name in f.name), None)
        if sns_file:
            sns_data = load_json(sns_file)
            sparsity = sns_data.get("average_sparsity_ratio")
        else:
            sparsity = None
            
        # Find corresponding GradFaith file
        gradfaith_file = next((f for f in gradfaith_files if dataset in f.name and model_name in f.name), None)
        if gradfaith_file:
            gradfaith_data = load_json(gradfaith_file)
            faithfulness = gradfaith_data.get("average_faithfulness")
        else:
            faithfulness = None
        
        # Get model size and family
        for config_name, config in models.items():
            if model_name.lower() in config_name.lower():
                model_size = config.get("params", None)
                break
        else:  # If no exact match found
            if "llama3-8b" in model_name or "llama-3-8b" in model_name:
                model_size = 8.0
            elif "llama3-70b" in model_name or "llama-3-70b" in model_name:
                model_size = 70.0
            elif "mixtral" in model_name.lower():
                model_size = 45.0
            elif "mistral" in model_name.lower():
                model_size = 7.0
            else:
                model_size = None
        
        # Determine model family (llama, gemma, etc.)
        if "scraft" in model_name.lower() or "scr-aft" in model_name.lower():
            model_family = "scraft"  # Special case for SCRaFT model
        else:
            model_family = next((name for name in ["llama", "gemma", "opt", "phi", "mistral", "mixtral"] 
                              if name in model_name.lower()), "other")
        
        # Store the data
        data.append({
            "model": model_name,
            "dataset": dataset,
            "accuracy": accuracy,
            "cst_rate": cst_data["sensitivity_rate"],
            "sparsity": sparsity,
            "faithfulness": faithfulness,
            "model_size": model_size,
            "model_family": model_family
        })
    
    # Convert to DataFrame
    return pd.DataFrame(data)

def plot_scaling_curve(df, output_dir):
    """
    Plot CST rate vs model size (log scale) with model families.
    
    Args:
        df: DataFrame with evaluation metrics
        output_dir: Directory to save plots
    """
    plt.figure(figsize=(10, 6))
    
    # Group by model family
    families = df["model_family"].unique()
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*']
    colors = sns.color_palette("tab10", len(families))
    
    # Plot each family as a separate line
    for i, family in enumerate(families):
        family_df = df[df["model_family"] == family]
        
        # Skip if only one point (can't draw a line)
        if len(family_df) <= 1 and family != "scraft":
            plt.scatter(
                family_df["model_size"], 
                family_df["cst_rate"], 
                label=family,
                marker=markers[i % len(markers)],
                s=100,
                color=colors[i]
            )
        elif family == "scraft":
            # Special handling for SCRaFT model (red star)
            plt.scatter(
                family_df["model_size"], 
                family_df["cst_rate"], 
                label="SCRaFT", 
                marker='*',
                s=300,
                color='red',
                zorder=10
            )
        else:
            # Plot connected line for family with multiple points
            plt.plot(
                family_df["model_size"], 
                family_df["cst_rate"], 
                marker=markers[i % len(markers)],
                label=family,
                color=colors[i]
            )
            plt.scatter(
                family_df["model_size"], 
                family_df["cst_rate"], 
                s=100,
                color=colors[i]
            )
    
    plt.xscale("log")
    plt.grid(True, alpha=0.3)
    plt.xlabel("Model Size (B parameters)", fontsize=14)
    plt.ylabel("Contradiction Sensitivity Rate", fontsize=14)
    plt.title("Scaling of Contradiction Sensitivity with Model Size", fontsize=16)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, "cst_scaling.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "cst_scaling.pdf"))
    plt.close()

def plot_pareto_frontier(df, output_dir):
    """
    Plot accuracy vs CST rate to visualize Pareto frontier.
    
    Args:
        df: DataFrame with evaluation metrics
        output_dir: Directory to save plots
    """
    plt.figure(figsize=(10, 6))
    
    # Group by model family
    families = df["model_family"].unique()
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*']
    colors = sns.color_palette("tab10", len(families))
    
    for i, family in enumerate(families):
        family_df = df[df["model_family"] == family]
        
        if family == "scraft":
            # Highlight SCRaFT model
            plt.scatter(
                family_df["cst_rate"], 
                family_df["accuracy"], 
                label="SCRaFT", 
                marker='*',
                s=300,
                color='red',
                zorder=10
            )
        else:
            plt.scatter(
                family_df["cst_rate"], 
                family_df["accuracy"], 
                label=family,
                marker=markers[i % len(markers)],
                s=100,
                color=colors[i]
            )
    
    plt.grid(True, alpha=0.3)
    plt.xlabel("Contradiction Sensitivity Rate", fontsize=14)
    plt.ylabel("Task Accuracy", fontsize=14)
    plt.title("Accuracy vs. Contradiction Sensitivity (Pareto Frontier)", fontsize=16)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, "accuracy_cst_pareto.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "accuracy_cst_pareto.pdf"))
    plt.close()

def plot_faithfulness_accuracy(df, output_dir):
    """
    Plot gradient faithfulness vs accuracy with model families.
    
    Args:
        df: DataFrame with evaluation metrics
        output_dir: Directory to save plots
    """
    # Filter rows where faithfulness is not None
    faith_df = df.dropna(subset=["faithfulness"])
    
    if len(faith_df) == 0:
        print("No faithfulness data available for plotting")
        return
        
    plt.figure(figsize=(10, 6))
    
    # Group by model family
    families = faith_df["model_family"].unique()
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*']
    colors = sns.color_palette("tab10", len(families))
    
    for i, family in enumerate(families):
        family_df = faith_df[faith_df["model_family"] == family]
        
        if family == "scraft":
            # Highlight SCRaFT model
            plt.scatter(
                family_df["faithfulness"], 
                family_df["accuracy"], 
                label="SCRaFT", 
                marker='*',
                s=300,
                color='red',
                zorder=10
            )
        else:
            plt.scatter(
                family_df["faithfulness"], 
                family_df["accuracy"], 
                label=family,
                marker=markers[i % len(markers)],
                s=100,
                color=colors[i],
                alpha=0.7
            )
    
    plt.grid(True, alpha=0.3)
    plt.xlabel("Gradient Faithfulness", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.title("Faithfulness vs Accuracy", fontsize=16)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, "faithfulness_accuracy.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "faithfulness_accuracy.pdf"))
    plt.close()

def generate_csv_summary(df, output_dir):
    """
    Generate CSV summary table for paper.
    
    Args:
        df: DataFrame with evaluation metrics
        output_dir: Directory to save CSV
    """
    # Group by model and average across datasets
    summary = df.groupby("model").agg({
        "accuracy": "mean",
        "cst_rate": "mean",
        "sparsity": "mean",
        "faithfulness": "mean",
        "model_size": "first",
        "model_family": "first"
    }).reset_index()
    
    # Sort by model size
    summary = summary.sort_values("model_size")
    
    # Save to CSV
    summary.to_csv(os.path.join(output_dir, "scaling_summary.csv"), index=False)
    
    # Also generate dataset-specific summaries
    for dataset in df["dataset"].unique():
        dataset_df = df[df["dataset"] == dataset]
        dataset_df.to_csv(os.path.join(output_dir, f"{dataset}_summary.csv"), index=False)
    
    # Create a more structured table for paper presentation
    paper_table = df.pivot_table(
        index=["model", "model_size", "model_family"], 
        columns="dataset", 
        values=["accuracy", "cst_rate", "sparsity"]
    )
    
    paper_table.to_csv(os.path.join(output_dir, "paper_table.csv"))

def main():
    """
    Main function for plotting script.
    """
    parser = argparse.ArgumentParser(description="Plot SCRaFT benchmarking results")
    parser.add_argument(
        "--model_list",
        type=str,
        default="models.json",
        help="Path to models.json config file"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="logs",
        help="Directory containing evaluation logs"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="plots",
        help="Directory to save plots"
    )
    
    args = parser.parse_args()
    
    # Collect results
    df = collect_results(args.input_dir, args.model_list)
    
    # Generate plots
    plot_scaling_curve(df, args.output_dir)
    plot_pareto_frontier(df, args.output_dir)
    plot_faithfulness_accuracy(df, args.output_dir)  # Add new faithfulness plot
    
    # Generate CSV summary
    generate_csv_summary(df, args.output_dir)
    
    print(f"Plots and tables saved to {args.output_dir}")
    print(f"Found {len(df)} model-dataset combinations")

if __name__ == "__main__":
    main()
