# Small-LLM-Sequential-Instruction-Tuning
Sequential Instruction Tuning of a Small LLM with Strong-Model Judge Evaluation

LLM Sequential Fine-Tuning & Catastrophic Forgetting

This repository contains the codebase and data for an NLP experiment demonstrating two-stage Curriculum Learning using QLoRA. We fine-tuned Phi-3.5-mini-instruct to perform strict JSON data extraction while retaining its general conversational abilities.

Repository Structure

```text
├── README.md                   # This file
├── REPORT.md                   # Full 5-page qualitative analysis and results
├── requirements.txt            # Python dependencies
├── config.json                 # Centralized hyperparameters and model configurations
├── prompts.json                # Editable prompt templates for all stages
├── generate_all_json_tasks.py  # Synthetic data generation via Imitation Learning
├── prepare_eval_data.py        # Alpaca dataset cleaning and splitting
├── train_stage1_alpaca.py      # QLoRA Stage 1 training script
├── train_stage2_json.py        # QLoRA Stage 2 training script
├── run_mass_inference.py       # Inference script for model evaluation
├── run_llm_judge.py            # Pairwise LLM-as-a-Judge evaluation script
├── calculate_metrics_final.py  # Automated metrics (ROUGE, BERTScore, Field-Level F1)
└── slurm_scripts/              # UTSA HPC batch scripts for launching jobs
```

Setup & Installation

Clone the repository and navigate to the directory.

Create a virtual environment and install dependencies:
```bash
conda create -n llm_env python=3.10
conda activate llm_env
pip install -r requirements.txt
```

Set your API keys as environment variables for the synthetic data generation and evaluation judge:
```bash
export UTSA_API_KEY="your_key_here"
export UTSA_BASE_URL="http://10.246.100.230/v1"
```

Reproduction Steps

1. Data Construction
Generate the 100-item synthetic JSON instruct dataset and the 100-item Alpaca evaluation split:
```bash
python generate_all_json_tasks.py
python prepare_eval_data.py
```

2. Stage 1 Training (Alpaca Alignment)
Submit the training job to the HPC cluster to train the base model on 51k general instructions.
```bash
sbatch slurm_scripts/submit_stage1.slurm
```

3. Stage 2 Training (JSON Curriculum)
Submit the Stage 2 job to continue fine-tuning the Stage 1 adapter exclusively on the synthetic JSON data.
```bash
sbatch slurm_scripts/submit_stage2.slurm
```

4. Inference & Evaluation
Generate responses across all three checkpoints and calculate quantitative metrics:
```bash
python run_mass_inference.py
python calculate_metrics_final.py
python run_llm_judge.py
```
