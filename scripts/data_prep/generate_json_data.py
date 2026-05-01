import json
import os
import random
import re
from openai import OpenAI
from tqdm import tqdm

API_KEY = os.getenv("OPENAI_API_KEY", "dummy-key")
BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# Load configs
with open("config/config.json", "r") as f:
    config = json.load(f)

with open("config/prompts.json", "r") as f:
    prompts = json.load(f)

TEACHER_MODEL = config["models"]["teacher_model"]
TARGET_TOTAL = config["data_generation"]["target_train_samples"] + config["data_generation"]["target_eval_samples"]

PROCESSED_TRAIN_PATH = "data/processed/json_train.json"
EVAL_PATH = "data/eval/json_eval.json"

def clean_json_string(raw_response):
    """Attempts to strip markdown blocks if the model ignored the system prompt."""
    raw_response = raw_response.strip()
    if raw_response.startswith("```json"):
        raw_response = raw_response[7:]
    if raw_response.startswith("```"):
        raw_response = raw_response[3:]
    if raw_response.endswith("```"):
        raw_response = raw_response[:-3]
    return raw_response.strip()

def generate_teacher_response(instruction, input_text):
    """Calls the teacher model and validates JSON output."""
    system_prompt = prompts["teacher_generation"]["system_prompt"]
    user_prompt = f"Instruction: {instruction}\nInput: {input_text}"

    try:
        response = client.chat.completions.create(
            model=TEACHER_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3, # Low temp for structured formats
            max_tokens=1024
        )
        
        raw_output = response.choices[0].message.content
        cleaned_output = clean_json_string(raw_output)
        
        # Validate JSON correctness
        json_obj = json.loads(cleaned_output)
        
        # If successful, return the stringified validated JSON to ensure perfect formatting
        return json.dumps(json_obj, indent=2)
    
    except Exception as e:
        # Fails if JSON is invalid or API drops
        return None

def main():
    print(f"Connecting to Teacher Model at {BASE_URL}...")
    
    # Load seed data (Alpaca) to give the model diverse context to work with
    with open("data/processed/alpaca_train.json", "r", encoding="utf-8") as f:
        seed_data = json.load(f)
    
    random.shuffle(seed_data)
    
    valid_samples = []
    task_types = ["extraction", "schema_constrained", "exact_label", "json_repair", "tool_call"]
    
    # Schemas and categories for dynamic prompt building
    schemas = ['{"title": "string", "summary": "string", "sentiment": "string"}', 
               '{"key_points": ["string"], "word_count": "integer"}']
    labels = ["['URGENT', 'ROUTINE', 'INFORMATIONAL']", "['POSITIVE', 'NEGATIVE', 'NEUTRAL']"]
    functions = ["schedule_meeting", "search_database", "send_email"]

    pbar = tqdm(total=TARGET_TOTAL, desc="Generating JSON Data")
    
    seed_idx = 0
    while len(valid_samples) < TARGET_TOTAL and seed_idx < len(seed_data):
        seed = seed_data[seed_idx]
        seed_idx += 1
        
        task_type = random.choice(task_types)
        base_text = seed["output"][:500] # Use the output of alpaca as an input for the teacher to process
        
        # Dynamically build the instruction based on the chosen task type
        if task_type == "extraction":
            instruction = prompts["teacher_generation"]["task_extraction"]
        elif task_type == "schema_constrained":
            instruction = prompts["teacher_generation"]["task_schema_constrained"].replace("{schema}", random.choice(schemas))
        elif task_type == "exact_label":
            instruction = prompts["teacher_generation"]["task_exact_label"].replace("{labels}", random.choice(labels))
        elif task_type == "json_repair":
            # Deliberately break a simple JSON
            broken_json = '{"text": "' + base_text[:50] + '", "status": missing_quotes}'
            instruction = prompts["teacher_generation"]["task_json_repair"]
            base_text = broken_json
        elif task_type == "tool_call":
            instruction = prompts["teacher_generation"]["task_tool_call"].replace("{function_name}", random.choice(functions))
            base_text = seed["instruction"] # Use the Alpaca instruction as a user command
        
        # Execute the prompt
        instruction_text = instruction.replace("{input}", "") # Clean placeholder
        
        validated_json = generate_teacher_response(instruction_text, base_text)
        
        if validated_json:
            valid_samples.append({
                "instruction": instruction_text,
                "input": base_text,
                "output": validated_json
            })
            pbar.update(1)
            
            # Save progress incrementally to avoid data loss
            if len(valid_samples) % 50 == 0:
                with open("data/processed/temp_json_generation.json", "w", encoding="utf-8") as f:
                    json.dump(valid_samples, f, indent=2)

    pbar.close()
    
    if len(valid_samples) < TARGET_TOTAL:
        print("Warning: Exhausted seed data before reaching target sample size.")
        
    print(f"Generated {len(valid_samples)} valid JSON samples.")
    
    # Split into Train and Eval
    eval_size = config["data_generation"]["target_eval_samples"]
    eval_data = valid_samples[:eval_size]
    train_data = valid_samples[eval_size:]
    
    os.makedirs(os.path.dirname(PROCESSED_TRAIN_PATH), exist_ok=True)
    
    with open(PROCESSED_TRAIN_PATH, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
        
    with open(EVAL_PATH, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False)
        
    # Clean up temp file
    if os.path.exists("data/processed/temp_json_generation.json"):
        os.remove("data/processed/temp_json_generation.json")
        
    print("Phase 1b Complete! JSON datasets saved.")

if __name__ == "__main__":
    main()