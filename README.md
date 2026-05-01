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

# 2. Experiments

### 4.1 & 4.2 Alpaca / General Tasks Comparison
| Checkpoint | Alpaca Judge Win Rate | R-1 | R-2 | ROUGE-L | BERTScore | Avg Length |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| CP0: Base | N/A (Baseline) | 28.2 | 10.8 | 17.0 | 85.0 | 450 words |
| CP1: Stage 1 | 26.3% vs CP0 | 27.8 | 11.3 | 17.4 | 85.2 | 487 words |
| CP2: Stage 2 | 47.5% vs CP1 | 27.6 | 11.1 | 17.0 | 84.9 | 479 words |

- Fine-tuning on Alpaca (Stage 1) left to a slight rise in ROUGE-L (17.0 → 17.4) and a consistent BERTScore (85.2).
- Average word count increased from 450 to 487 words, showing the model learned to give more detailed answers.
- CP1 only won 26.3% of the time against the Base model.  It's possible that the there is a verbosity bias with the large judge model.  

### 4.3 JSON Structured Output Evaluation
| Checkpoint | JSON Judge Win Rate | JSON Validity | Schema Compliance | Exact Match | Field-Level F1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| CP0: Base | N/A (Baseline) | 2.0% | 100.0% | 0.0% | 0.0% |
| CP1: Stage 1 | N/A | 1.0% | 100.0% | 0.0% | 0.0% |
| CP2: Stage 2 | 28.2% vs CP1 | 2.0% | 100.0% | 0.0% | 0.0% |

- Strict programmatic metrics (Exact Match, Field-Level F1) remained at 0%, and JSON validity barely moved (staying around 2.0%).
- It's likely that the model still tried to act like a chatbot, not adhering to the strict JSON format that was expected.  This can break the Python json.loads() parser.
- However, of the 2.0% of outputs that did parse correctly, Schema Compliance was 100%.

### 4.4 Forgetting Analysis
- Comparing Alpaca CP1 and CP2, the model's performance on general tasks did not have much difference.
- ROUGE-L only changed by .4 points (17.4 to 17.0).
- BERTScore only dropped by .3 points (85.2 to 84.9).  The semantic quality of the text was mostly preserved.
- The Llama 70B judge actually preferred CP2 over CP1 47.5% of the time, showing that it retained assistant qualities.
- Based on purely qualitative metrics, catastrophic forgetting did not occur.

### 4.5 Ablation Study: Stage 2 Training Epochs
| Stage 2 Epochs | JSON Validity (Capability Gained) | Alpaca ROUGE-L (Capability Retained) |
| :--- | :--- | :--- |
| 1 Epoch | 1.0% | 15.2 |
| 2 Epochs | 2.0% | 15.0 |
| 3 Epochs (Main) | 2.0% | 17.0 |

- We tested 1, 2, and 3 epochs during Stage 2 to see how training duration affected capability gained (JSON) vs. capability retained (Alpaca ROUGE-L).
- Scaling the epochs didn't really help the JSON validity, which stayed around 1-2 percent.  It didn't fix the chattiness.
- There was a small drop in ROUGE-L at 1 and 2 epochs (down to 15.0 by .2).  Stopping too early appeared to weaken the model.  However, 3 epochs allowed for some stabalization back up to 17.

### 4.6 Figures

# Stage 1 Loss

<img width="2400" height="1500" alt="stage1_loss" src="https://github.com/user-attachments/assets/09b7af43-b1e3-45a9-8c2a-4f6755fe69ea" />

# Stage 2 Loss

<img width="2400" height="1500" alt="stage2_loss" src="https://github.com/user-attachments/assets/4980c9c8-069e-4817-af17-599569791439" />

# Judge Winning Rates

<img width="2700" height="1800" alt="judge_win_rates" src="https://github.com/user-attachments/assets/2177f217-eba1-4927-bdb3-7016cb47693c" />

# 3. Analysis

## Qualitative Comparison Across Checkpoints:

- **CP0 (Base)**: This primarily acted like a standard conversational model.  It was chatty and prone to rambling, and would often ignore the strict formatting constraints.
- **CP1 (Alapaca Tuned)**: This had a notable improvement in answering general questions directly, however it still often failed at producing well formatted JSON outputs (0% exact match).
- **CP2 (JSON Tuned)**: This model actively tried to use JSON keys and brackets, but still clung to its base conversational nature, such as "Here is the JSON you requested", which often broke the strict parser.

## Failure Case Analysis

- The 2% JSON validity very much stood out.  While the model did learn JSON structure (given the 100% Schema Compliance when valid), it failed the strict Python json.loads() parser often due to conversational filler.
- The 1000 samples of Supervised Fine-Tuning (SFT) does not appear to be enough to overwrite the conversational nature of the small model.  Ideally, the SFT would be paired with some constrained decoding libraries to force validity.

## Forgetting vs Retention

- Based on the results, it showed that catastrophic forgetting did not occur.  It was able to retain it's original conversational nature.
- ROUGE-L stayed stable at  around 17.0 and the Llama 70B judge actually preferred CP2 over CP1 47.5% of the time.

## Implications

- The Stage 2 JSON dataset was generated by a highly capable teacher (Llama-3.3-70B).  However, since the text INSIDE the JSON values contained various complex language, reasoning, and grammar, it might have detracted from the overall JSON structure.  It likely reinforced the conversational capabilities while trying to teach syntax at the same time.  In further attempts, I would make the larger model be far more concise in how it teaches.

# 4. Prompt Engineering

**Teacher Generation Prompts (Imitation Learning):**

- The main idea was to make the prompts as concise and structured as possible.  This was to reduce the inclusion of markdown and chatter that would break the training. 
> "ONLY output valid JSON. Do not include markdown formatting like ```json. Do not include introductory text."
- We dynamically passed the 5 required tasks (schema constrained, extraction, exact label, repair, tool call) into the prompt using Python variables ({schema}, {labels}, etc.), to further constrain the prompt.

**Judge Evaluation Prompts:**
- The judge prompt was specifically designed to output its own strict JSON format so the Python script could parse the scores automatically.
- We explicitely defined the 6 evaluation criteria (Instruction Following, Correctness, Clarity, Completeness, Structured Output, Hallucination) to force the Llama 70B judge to justify its answers.
- To help combat the "positional bias" of the LLM, where judges appear to prefer the first response, the python script would randomize wheather CP1 or CP2 was placed in the Response A slot.

**Prompt Failures:**
- **Judge Explanations Breaking the Parser:** Initially, the judge would output it's JSON scorecard, but then it would add a paragraph explaining its thoughts.  This would break the autoamtic metric calculation.  To fix this, I added another instruction to place this explanation inside the JSON itself (Reasoning), so it would be parsable.
- **Positional Bias:** Earlier runs showed that the judge skewed toward the first response (Response A) because it was read first.  I was able to get around this by programmatically randomizing the response order.  I would also have the judge calculate the scores for all 6 metrics first, before it was allowed to declare a "winner". 

# Appendix

## Prompts:

```
{
  "student_training_schema": {
    "format": "Instruction: {instruction}\nInput: {input}\nOutput: {output}"
  },
  "teacher_generation": {
    "system_prompt": "You are an expert data generator. Your task is to produce high-quality, valid JSON outputs based on the provided instruction.",
    "task_extraction": "Extract the key entities, dates, and attributes from the following text into a JSON object. Text: {input}",
    "task_schema_constrained": "Generate a valid JSON object that strictly conforms to the following schema: {schema}. Context: {input}",
    "task_exact_label": "Classify the following text and return the result as a JSON object with the key 'classification' and one of the following allowed labels: {labels}. Text: {input}",
    "task_json_repair": "Fix the following malformed JSON into valid JSON. Malformed JSON: {input}",
    "task_tool_call": "Produce a JSON object representing a function call to '{function_name}' with the appropriate named parameters based on the following request: {input}"
  },
  "judge_evaluation": {
    "system_prompt": "You are an impartial judge evaluating two responses from an AI model. Evaluate them based on Instruction Following, Correctness, Clarity, Completeness, Structured Output Validity, and Hallucination Risk.",
    "prompt_template": "Prompt ID: {prompt_id}\nInstruction: {instruction}\n\nResponse A:\n{response_a}\n\nResponse B:\n{response_b}\n\nProvide your evaluation in the following strict JSON schema:\n{\n  \"prompt_id\": \"{prompt_id}\",\n  \"checkpoint_a\": \"{checkpoint_a}\",\n  \"checkpoint_b\": \"{checkpoint_b}\",\n  \"response_a_scores\": {\n    \"instruction_following\": <1-5>,\n    \"correctness\": <1-5>,\n    \"clarity\": <1-5>,\n    \"completeness\": <1-5>,\n    \"structured_output_validity\": <1-5>,\n    \"hallucination_risk\": <1-5>\n  },\n  \"response_b_scores\": {\n    \"instruction_following\": <1-5>,\n    \"correctness\": <1-5>,\n    \"clarity\": <1-5>,\n    \"completeness\": <1-5>,\n    \"structured_output_validity\": <1-5>,\n    \"hallucination_risk\": <1-5>\n  },\n  \"winner\": \"<A or B or Tie>\",\n  \"justification\": \"<short explanation>\"\n}"
  }
}
``` 

## Config

```
{
  "models": {
    "student_model": "microsoft/Phi-3.5-mini-instruct",
    "teacher_model": "llama-3.3-70b-instruct-awq",
    "judge_model": "llama-3.3-70b-instruct-awq"
  },
  "training": {
    "fine_tuning_method": "QLoRA",
    "precision": "4-bit",
    "learning_rate": 2e-5,
    "max_sequence_length": 2048,
    "batch_size": 4,
    "stage_1_epochs": 3,
    "stage_2_epochs": 3
  },
  "lora": {
    "rank": 16,
    "alpha": 32,
    "dropout": 0.05
  },
  "evaluation": {
    "alpaca_eval_prompts_count": 100,
    "json_eval_prompts_count": 100
  },
  "data_generation": {
    "target_train_samples": 1000,
    "target_eval_samples": 100
  }
}
```

# Overly Chatty Training Example:

```
  {
    "prompt_id": "prompt_1",
    "instruction": "What do you understand by AI explainability?",
    "input": "",
    "expected_output": "AI explainability refers to the ability to understand and interpret the reasoning and decision-making processes of artificial intelligence systems. It is an essential aspect of building transparent, fair, and trustworthy AI models, as it allows developers, stakeholders, and end-users to gain insights into how an AI system reaches its conclusions, and how it may be improved or refined to better meet intended goals.\n\nIn machine learning-based models, AI explainability can involve examining the weights, biases, and rules that the machine learning algorithm has learned and understanding how the model makes predictions based on input data. In deep learning-based models, the problem of explainability is even more challenging as these models can have millions of parameters and can behave like a 'black box', making it difficult to determine how they make decisions.\n\nExplainability is crucial for building accountable and transparent AI systems and for gaining users' trust. It is also important for regulatory compliance, as some regulations require companies to provide explanations for automated decisions made by AI systems.\n\nIn summary, AI explainability is the ability to interpret and describe the internal workings and decision-making processes of an AI system, making it transparent, trustworthy, and accountable to its users.",
    "generated_output": "Response:\nOutput: AI explainability refers to the ability to describe how artificial intelligence (AI) systems make decisions or arrive at conclusions. It's a critical aspect of AI transparency and accountability, ensuring that the processes behind AI-driven outcomes are understandable to humans. This involves clear communication of the logic, data, and algorithms that AI models use to process information and generate results.\n\nExplainability is important for several reasons:\n\n1. **Trust**: When users and stakeholders understand how AI systems work, they are more likely to trust and accept their outputs.\n\n2. **Regulatory Compliance**: In many industries, regulations require that automated systems be transparent and explainable, especially when they impact individuals' rights or well-being.\n\n3. **Error Reduction**: Understanding AI decisions can help identify and correct errors or biases in the system, leading to more accurate and fair outcomes.\n\n4. **Improvement**: Explainability allows developers to refine and improve AI models by understanding their strengths and weaknesses.\n\n5. **Ethical Considerations**: It's crucial for ensuring that AI systems operate within ethical boundaries, particularly in sensitive areas like healthcare, finance, and law enforcement.\n\nMethods for achieving AI explainability include:\n\n- **Feature Importance**: Techniques that identify which features in the data are most influential in the model's decision-making process.\n- **Model-agnostic Methods**: Tools that can be applied to any machine learning model to provide insights into its predictions.\n- **Local Interpretable Model-agnostic Explanations (LIME)**: Approaches that approximate the model locally with an interpretable one to explain individual predictions.\n- **SHapley Additive exPlanations (SHAP)**: A method based on game theory that assigns each feature an importance value for a particular prediction.\n- **Counterfactual Explanations**: Describing the smallest change needed in input features to change the outcome, which helps in understanding the decision boundary.\n\nExplainability is an ongoing area of research in AI, as it's challenging to balance complexity and interpretability, especially in complex models like deep neural networks."
  }
```



=======================================================

# Repository Structure

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
