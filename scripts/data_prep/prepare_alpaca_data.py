import json
import os
import random
from datasets import load_dataset

# Set random seed for reproducibility
random.seed(42)

# File paths based on our new directory structure
PROCESSED_TRAIN_PATH = "data/processed/alpaca_train.json"
EVAL_PATH = "data/eval/alpaca_eval.json"

def prepare_alpaca_data():
    print("Loading 'yahma/alpaca-cleaned' dataset...")
    # Load the dataset from Hugging Face
    dataset = load_dataset("yahma/alpaca-cleaned", split="train")
    
    cleaned_data = []
    malformed_count = 0

    print("Normalizing and validating schema...")
    for item in dataset:
        # Ensure all fields exist, defaulting to empty strings if missing
        instruction = item.get("instruction", "").strip()
        input_text = item.get("input", "").strip()
        output_text = item.get("output", "").strip()

        # Validate format consistency: skip if core components are completely empty
        if not instruction or not output_text:
            malformed_count += 1
            continue

        # Normalize into the unified training schema
        cleaned_data.append({
            "instruction": instruction,
            "input": input_text,
            "output": output_text
        })

    print(f"Removed {malformed_count} malformed or incomplete samples.")
    print(f"Total valid samples: {len(cleaned_data)}")

    # Shuffle the data to ensure a randomized distribution
    random.shuffle(cleaned_data)

    # Set aside exactly 100 samples for the held-out evaluation subset
    eval_size = 100
    eval_data = cleaned_data[:eval_size]
    train_data = cleaned_data[eval_size:]

    # Ensure directories exist
    os.makedirs(os.path.dirname(PROCESSED_TRAIN_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(EVAL_PATH), exist_ok=True)

    # Save the training set
    with open(PROCESSED_TRAIN_PATH, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    # Save the held-out evaluation set
    with open(EVAL_PATH, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False)

    print(f"\nSuccess!")
    print(f"Saved {len(train_data)} training samples to {PROCESSED_TRAIN_PATH}")
    print(f"Saved {len(eval_data)} evaluation samples to {EVAL_PATH}")

if __name__ == "__main__":
    prepare_alpaca_data()