# Small-LLM-Sequential-Instruction-Tuning
Sequential Fine-Tuning of a Small LLM with a Large LLM and Large LLM Judge Evaluation

This repository contains the codebase and data for an NLP experiment demonstrating two-stage Curriculum Learning using QLoRA. We fine-tuned Phi-3.5-mini-instruct to perform strict JSON data extraction while retaining its general conversational abilities.

# 1. Methodology
## 1.1 Model Selection
**Student Model:** microsoft/Phi-3.5-mini-instruct
>(3.8B parameters). This model was selected because it represents a state-of-the-art in the small model. Its compact size makes it highly suitable for single GPU QLoRA fine-tuning on the HPC, while its dense architecture provides a strong enough baseline to meaningfully measure both instruction adherence and structured output formatting.
>
**Teacher & Judge Model** Llama-3.3-70B-Instruct
>(accessed via local VPN API). A large model was required both to generate high quality synthetic training data (imitation learning) and to serve as a judge for qualitative evaluation.

## 1.2 Data Construction & Imitation Learning Pipeline
The training pipeline used two datasets:
>**Stage 1 (General Instruction Data):** We used a cleaned variant of the standard Alpaca dataset consisting of roughly 51,500 instruction-input-output pairs. 100 samples were held out strictly for evaluation.
>
>**Stage 2 (Structured JSON Instruct Data):** We constructed a synthetic dataset of 1,000 samples using imitation learning (black-box distillation). Prompts were designed across five required task types: JSON extraction, schema-constrained generation, exact-label classification, JSON repair, and tool-call argument generation. These prompts were fed to the Llama 70B teacher model. All teacher outputs were strictly validated for JSON parseability before being paired with their prompts to form the Stage 2 training targets. 100 samples were held out for evaluation.

## 1.3 UTSA HPC Setup & Training Design
>Training was executed on the UTSA ARC cluster using gpu1a100 (NVIDIA A100 GPU) managed via Slurm.
>
>The sequential fine-tuning process was conducted using 4-bit QLoRA (Quantized Low-Rank Adaptation) to maximize memory efficiency. The base model was loaded in nf4 quantization with bfloat16 compute precision.
>
>**Stage 1 (Alpaca):** The base Phi-3.5 model was fine-tuned on the Alpaca dataset to establish a strong general instruction capability.
>
>**Stage 2 (Teacher JSON):** The adapter weights from Stage 1 were loaded, unfrozen, and continuously fine-tuned on the synthetic JSON dataset.

## Hyperparameters (Consistent across both stages):
**LoRA Configuration:**
- Rank (r) = 16, 
- Alpha (α) = 32, 
- Dropout = 0.05 

Target modules included all linear layers (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj).

**Training:** 
- Learning Rate = 2e-5 with a cosine scheduler 
- 0.03 warmup ratio
- Max sequence length was constrained to 1024 tokens

**Epochs:** 
- Stage 1 trained for 3 epochs.
- Stage 2 trained for 3 epochs (with an ablation study conducting parallel runs at 1 and 2 epochs)

## 1.4 Evaluation Protocol
To isolate the effects of sequential training and measure potential catastrophic forgetting, the model was evaluated at three distinct checkpoints:
- Checkpoint 0 (CP0): The untuned Phi-3.5 base model.
- Checkpoint 1 (CP1): After Stage 1 Alpaca fine-tuning.
- Checkpoint 2 (CP2): After Stage 2 JSON fine-tuning.

At each checkpoint, the model generated responses for both the 100 Alpaca and 100 JSON held-out evaluation sets using very low temperature sampling (temperature=0.1).

## Metrics & LLM Judge:
**General Instruction (Alpaca):** 
Evaluated using ROUGE (1, 2, L), BERTScore, and a pairwise LLM Judge based on the Self-Instruct methodology. The Llama 70B judge compared CP0 vs. CP1, and CP1 vs. CP2, randomizing response order to prevent positional bias. The judge scored pairs across six criteria: Instruction Following, Correctness, Clarity, Completeness, Structured Output Validity, and Hallucination Risk.

**Structured Output (JSON):** Evaluated using strict metrics, including exact match, JSON parsing validity, Schema Compliance (key matching), and F1 scores (precision/recall of key-value pairs).






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
│   │   ├── judge_cp0_vs_cp1.json # Judge score files
│   │   ├── judge_cp1_vs_cp2.json 
│   │   ├── judge_cp1_vs_cp2_json.json
│   └── run_logs # Log outputs from scripts 
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
    │   ├── calculate_ablation_metrics.py
    │   ├── run_ablation_inference.py
    │   └── calculate_metrics_final.py
    │
    └── slurm/
        ├── run_stage1.slurm
        ├── run_stage2.slurm
        ├── run_ablation_inference.slurm
        ├── run_inference.slurm
        └── run_stage2_ablation.slurm
```

=======================================================

## REPRODUCTION & SETUP STEPS

### --- MODELS USED ---
```
Student Model: "microsoft/Phi-3.5-mini-instruct"
Teacher/Judge Model: "llama-3.3-70b-instruct-awq" (Via VPN API)
```

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
