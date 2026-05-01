# Small-LLM-Sequential-Instruction-Tuning
Sequential Fine-Tuning of a Small LLM with a Large LLM and Large LLM Judge Evaluation

This repository contains the codebase and data for an NLP experiment demonstrating two-stage Curriculum Learning using QLoRA. We fine-tuned Phi-3.5-mini-instruct to perform strict JSON data extraction while retaining its general conversational abilities.

Repository Structure

```text
LLM_Training_Project/
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
│   ├── judge_results/    
│       ├── judge_cp0_vs_cp1.json # Judge score files
│       ├── judge_cp1_vs_cp2.json 
│       ├── judge_cp1_vs_cp2_json.json
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

=======================================================

## REPRODUCTION & SETUP STEPS

### --- MODELS USED ---
Student Model: "microsoft/Phi-3.5-mini-instruct"

Teacher/Judge Model: "llama-3.3-70b-instruct-awq" (Via VPN API)

=======================================================

### PHASE 1: LOCAL DATA PREPARATION & IMITATION LEARNING

Set up the local environment and API variables (Windows):
```
setx OPENAI_API_KEY "your_api_key_here"
setx OPENAI_BASE_URL "http://your-vpn-ip:port/v1"
```

Install local dependencies:
```
pip install openai tqdm
```

Process the Alpaca Instruction Dataset:
```
python scripts/data_prep/prepare_alpaca_data.py
-> CREATED: data/processed/alpaca_train.json
-> CREATED: data/eval/alpaca_eval.json
```

Generate Synthetic JSON Curriculum via Teacher Model:
```
python scripts/data_prep/generate_json_data.py
-> CREATED: data/processed/json_train.json
-> CREATED: data/eval/json_eval.json
```

=======================================================

### PHASE 2: ARC CLUSTER ENVIRONMENT SETUP (UTSA HPC)

Connect to the ARC cluster and load Anaconda:
```
module load anaconda3
```

Create and activate a fresh environment:
```
conda create --prefix /work/UTSA_ID/llm_assignment3_env python=3.10 -y
conda activate /work/UTSA_ID/llm_assignment3_env
```

Install PyTorch (CUDA 12.1) and Hugging Face Ecosystem:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate trl bitsandbytes scipy tqdm wandb
```

Downgrade PEFT to fix float8/bfloat16 compatibility bug:
```
pip install peft==0.13.2
```
=======================================================

### PHASE 3: MODEL TRAINING ON ARC

(Note: Before running slurm scripts transferred from Windows, always run dos2unix on them)
```
dos2unix scripts/slurm/*.slurm
```

Run Stage 1 (Alpaca QLoRA Tuning):
```
sbatch scripts/slurm/run_stage1.slurm
-> CREATED: models/stage1_adapter/
```

Run Stage 2 (JSON Curriculum Tuning):
```
sbatch scripts/slurm/run_stage2.slurm
-> CREATED: models/stage2_adapter/
```

Run Stage 2 Epoch Ablation Study:
```
sbatch scripts/slurm/run_stage2_ablation.slurm
-> CREATED: models/stage2_adapter_1ep/
-> CREATED: models/stage2_adapter_2ep/
```

=======================================================

### PHASE 4: MASS INFERENCE ON ARC

Generate responses for Main Checkpoints (CP0, CP1, CP2):
```
sbatch scripts/slurm/run_inference.slurm
-> CREATED: logs/inference_results/cp0_alpaca_results.json
-> CREATED: logs/inference_results/cp0_json_results.json
-> CREATED: logs/inference_results/cp1_alpaca_results.json
-> CREATED: logs/inference_results/cp1_json_results.json
-> CREATED: logs/inference_results/cp2_alpaca_results.json
-> CREATED: logs/inference_results/cp2_json_results.json
```

Generate responses for Ablation Adapters:
```
sbatch scripts/slurm/run_ablation_inference.slurm
-> CREATED: logs/inference_results/cp2_1ep_alpaca_results.json
-> CREATED: logs/inference_results/cp2_1ep_json_results.json
-> CREATED: logs/inference_results/cp2_2ep_alpaca_results.json
-> CREATED: logs/inference_results/cp2_2ep_json_results.json
```

=======================================================

### PHASE 5: LOCAL EVALUATION & METRICS

(Transfer all JSON files from ARC logs/inference_results/ to local machine)

Install local NLP evaluation metrics:
```
pip install rouge-score bert-score
```

Run LLM-as-a-Judge (Pairwise Evaluation via VPN):
```
python scripts/evaluation/run_llm_judge.py
-> CREATED: logs/judge_cp0_vs_cp1.json
-> CREATED: logs/judge_cp1_vs_cp2.json
-> CREATED: logs/judge_cp1_vs_cp2_json.json
```

Calculate Final Rubric Metrics (ROUGE, BERTScore, JSON Validity):
```
python scripts/evaluation/calculate_metrics_final.py
```

Calculate Ablation Metrics:
```
python scripts/evaluation/calculate_ablation_metrics.py
```
