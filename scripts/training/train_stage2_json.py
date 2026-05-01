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
from peft import PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer

def main():
    print("Initializing Stage 2: JSON Instruct Curriculum Fine-Tuning...")

    # 1. Load Configurations
    with open("config/config.json", "r") as f:
        config = json.load(f)
    
    with open("config/prompts.json", "r") as f:
        prompts = json.load(f)

    # Extract hyperparams
    MODEL_ID = config["models"]["student_model"]
    LR = config["training"]["learning_rate"]
    BATCH_SIZE = config["training"]["batch_size"]
    EPOCHS = config["training"]["stage_2_epochs"]
    MAX_SEQ_LENGTH = config["training"]["max_sequence_length"]
    
    STAGE1_ADAPTER = "models/stage1_adapter"
    OUTPUT_DIR = "models/stage2_adapter"

    # 2. Load the synthetic JSON Dataset generated in Phase 1b
    print("Loading teacher-generated JSON Instruct data...")
    dataset = load_dataset("json", data_files="data/processed/json_train.json", split="train")

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

    # 3. Model Setup (4-bit QLoRA)
    print(f"Loading base {MODEL_ID} in 4-bit precision...")
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

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    base_model = prepare_model_for_kbit_training(base_model)

    # 4. Load Stage 1 Adapter & Unfreeze it
    print(f"Loading Stage 1 adapter from {STAGE1_ADAPTER} and enabling training...")
    model = PeftModel.from_pretrained(base_model, STAGE1_ADAPTER, is_trainable=True)
    model.print_trainable_parameters()

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
        bf16=True, 
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        report_to="none" 
    )

    # 6. Trainer Initialization
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        formatting_func=formatting_prompts_func,
        processing_class=tokenizer,
        args=training_args,
    )

    # 7. Execute Training
    print("Starting Stage 2 Training on JSON Data...")
    trainer.train()

    # 8. Save Final Adapter
    print(f"Saving final Stage 2 adapter to {OUTPUT_DIR}...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Stage 2 Curriculum Training Complete!")

if __name__ == "__main__":
    main()