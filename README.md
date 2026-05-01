# Small-LLM-Sequential-Instruction-Tuning
Sequential Fine-Tuning of a Small LLM with a Large LLM and Large LLM Judge Evaluation

This repository contains the codebase and data for an NLP experiment demonstrating two-stage Curriculum Learning using QLoRA. We fine-tuned Phi-3.5-mini-instruct to perform strict JSON data extraction while retaining its general conversational abilities.

Repository Structure

```text
llm_assignment3/
│
├── README.md                 # Blog post & Setup instructions
├── requirements.txt          # (peft, trl, transformers, datasets, rouge-score, bert-score, openai)
│
├── config/
│   ├── config.json           # Model names, LR, epochs, batch size, max tokens, LoRA params
│   └── prompts.json          # Editable templates (Teacher, Student, Judge)
│
├── data/
│   ├── processed/            # alpaca_train.json, json_train.json
│   └── eval/                 # alpaca_eval.json, json_eval.json
│
├── logs/
│   ├── inference_results/    # The six generated CP0/CP1/CP2 .json outputs
│   ├── judge_cp0_vs_cp1.json # Judge score files
│   ├── judge_cp1_vs_cp2.json 
│   ├── judge_cp1_vs_cp2_json.json
│   └── stage1_training.out   
│
└── scripts/
    ├── data/              
    │   ├── prepare_alpaca_data.py
    │   └── generate_json_data.py 
    │
    ├── training/
    │   ├── train_stage1_alpaca.py
    │   ├── train_stage2_json.py
    │   └── train_stage2_json_ablation.py
    │
    ├── evaluation/
    │   ├── run_mass_inference.py
    │   ├── run_llm_judge.py
    │   └── calculate_metrics_final.py
    │
    └── slurm/
        ├── run_stage1.slurm
        └── run_stage2.slurm
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
