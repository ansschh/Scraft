# SCRaFT: Self-Contradiction Regularised Fine-Tuning

This repository implements SCRaFT (Self-Contradiction Regularised Fine-Tuning), a method to teach large language models to produce more faithful chain-of-thought explanations by making them fear contradictions in their own reasoning.

## Problem Statement

Large-language-model "chain-of-thought" (CoT) prompting often improves task accuracy while simultaneously producing explanations that the model itself does *not* rely on. If you delete, reorder, or even contradict big chunks of the chain, the answer usually stays unchanged. 

That *decorative reasoning* is dangerous: 
- It hides spurious shortcuts
- Blocks error analysis
- Undermines any downstream safety tooling that assumes the chain is trustworthy

## Our Approach

Instead of translating the chain into symbolic code (heavy engineering) or merely measuring how unfaithful it is (no cure), we teach an **open-weight** model to fear contradictions in its own reasoning:

1. **Inject** one logically incompatible sentence into its freshly generated CoT.
2. **Ask the model itself to fix that contradiction.**
3. **Fine-tune** with a *pairwise KL / Direct-Preference* loss so that the *corrected* chain is preferred over the *contradicted* one.

The result: shorter, contradiction-sensitive explanations **without** hurting task accuracy—and the entire pipeline runs in a few GPU-hours.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/scaft.git
cd scaft

# Create conda environment
conda env create -f env.yml
conda activate scaft

# Authenticate with Hugging Face (needed for gated models like Llama-3)
huggingface-cli login
```

## Repository Structure

- `env.yml`: Conda environment specification with locked dependencies
- `src/`: Main source code directory
  - `00_generate_cot.py`: Generates vanilla chain-of-thought reasoning
  - `01_inject_contradiction.py`: Mutates chains by flipping facts
  - `02_self_refine.py`: Prompts the model to repair its own contradictions
  - `03_build_pairs.py`: Converts data into DPO preference pairs
  - `04_finetune_scr_aft.py`: Runs DPO fine-tuning with LoRA adapters
  - `05_eval_accuracy.py`: Evaluates task accuracy on new chains
  - `06_eval_cst.py`: Contradiction Sensitivity Test
  - `07_eval_sns.py`: Sentence-Necessity Search
  - `08_eval_gradfaith.py`: Computes Gradient-Faithfulness metric

## Usage Pipeline

### 1. Generate Original Chains

```bash
python src/00_generate_cot.py --model meta-llama/Meta-Llama-3-8B-Instruct --dataset aqua_rat --samples 500
```

### 2. Inject Contradictions

```bash
python src/01_inject_contradiction.py --input_file logs/aqua_rat_cot.json
```

### 3. Self-Repair Contradictions

```bash
python src/02_self_refine.py --model meta-llama/Meta-Llama-3-8B-Instruct --input_file logs/aqua_rat_cot_contradicted.json
```

### 4. Build DPO Preference Pairs

```bash
python src/03_build_pairs.py --input_file logs/aqua_rat_cot_contradicted_repaired.json
```

### 5. Fine-Tune with SCRaFT

```bash
python src/04_finetune_scr_aft.py --model meta-llama/Meta-Llama-3-8B-Instruct --dataset_path logs/aqua_rat_cot_contradicted_repaired_dpo_dataset
```

### 6. Evaluate Results

```bash
# Evaluate accuracy
python src/05_eval_accuracy.py --model ft-scraft/adapter --dataset aqua_rat

# Evaluate contradiction sensitivity
python src/06_eval_cst.py --model ft-scraft/adapter --cache logs/predict_cache.pkl

# Evaluate sentence necessity
python src/07_eval_sns.py --model ft-scraft/adapter --cache logs/predict_cache.pkl

# Evaluate gradient faithfulness
python src/08_eval_gradfaith.py --model ft-scraft/adapter --cache logs/predict_cache.pkl
```

## Datasets

This implementation supports:
- **AQuA-RAT**: Algebraic word problems
- **ARC-Easy**: Multiple-choice science questions

Additional datasets can be integrated by modifying the data loading functions in the scripts.

## Key Features

1. **End-to-End Pipeline**: Complete process from generating chains to evaluation
2. **Efficient Implementation**: Uses vLLM for high-throughput generation
3. **Parameter-Efficient Fine-Tuning**: LoRA adapters for minimal compute requirements
4. **Multiple Evaluation Metrics**: Triangulates improvements with complementary metrics
5. **Fully Reproducible**: Seeds fixed for deterministic results

## Citation

If you use this code in your research, please cite:

```bibtex
@article{scraft2025,
  title={SCRaFT: Self-Contradiction Regularised Fine-Tuning for Faithful Chain-of-Thought Reasoning},
  author={Your Name},
  journal={arXiv preprint},
  year={2025}
}
```

## License

This project is licensed under the Apache License 2.0. See the LICENSE file for details.
