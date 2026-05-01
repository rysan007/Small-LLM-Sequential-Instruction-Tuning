import json
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

def main():
    print("Initializing Stage 1: Alpaca QLoRA Fine-Tuning...")

    # 1. Load Configurations
    with open("config/config.json", "r") as f:
        config = json.load(f)
    
    with open("config/prompts.json", "r") as f:
        prompts = json.load(f)

    # Extract hyperparams
    MODEL_ID = config["models"]["student_model"]
    LR = config["training"]["learning_rate"]
    BATCH_SIZE = config["training"]["batch_size"]
    EPOCHS = config["training"]["stage_1_epochs"]
    MAX_SEQ_LENGTH = config["training"]["max_sequence_length"]
    OUTPUT_DIR = "models/stage1_adapter"

    # 2. Load Dataset
    print("Loading pre-processed Alpaca training data...")
    dataset = load_dataset("json", data_files="data/processed/alpaca_train.json", split="train")

    # Formatting function using the schema from prompts.json
    template = prompts["student_training_schema"]["format"]
    def formatting_prompts_func(example):
        # Handle batch mode (list of strings)
        if isinstance(example['instruction'], list):
            output_texts = []
            for i in range(len(example['instruction'])):
                text = template.replace("{instruction}", example['instruction'][i]) \
                               .replace("{input}", example['input'][i]) \
                               .replace("{output}", example['output'][i])
                output_texts.append(text)
            return output_texts
        # Handle single-row mode (single string)
        else:
            text = template.replace("{instruction}", example['instruction']) \
                           .replace("{input}", example['input']) \
                           .replace("{output}", example['output'])
            return text

    # 3. Model & Tokenizer Setup (4-bit QLoRA)
    print(f"Loading {MODEL_ID} in 4-bit precision...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = MAX_SEQ_LENGTH

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    model = prepare_model_for_kbit_training(model)

    # 4. LoRA Configuration
    peft_config = LoraConfig(
        r=config["lora"]["rank"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        learning_rate=LR,
        logging_steps=10,
        num_train_epochs=EPOCHS,
        save_strategy="epoch",
        optim="paged_adamw_32bit",
        bf16=True, # A100 supports bf16
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        report_to="none" 
    )

    # 6. Trainer Initialization
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        formatting_func=formatting_prompts_func,
        processing_class=tokenizer,
        args=training_args,
    )

    # 7. Execute Training
    print("Starting Stage 1 Training...")
    trainer.train()

    # 8. Save Final Adapter
    print(f"Saving final Stage 1 adapter to {OUTPUT_DIR}...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Stage 1 Training Complete!")

if __name__ == "__main__":
    main()