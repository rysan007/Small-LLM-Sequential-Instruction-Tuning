import json
import os
import torch
import gc
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

def generate_responses(model, tokenizer, dataset, prompt_template):
    """Generates responses for a given dataset using the provided model."""
    results = []
    for i, item in enumerate(tqdm(dataset, desc="Generating")):
        # Format the prompt using the training schema, leaving the output blank for the model to complete
        formatted_prompt = prompt_template.replace("{instruction}", item.get("instruction", "")) \
                                          .replace("{input}", item.get("input", "")) \
                                          .replace("{output}", "")
        
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.1, # Low temp for deterministic evaluation
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
        # Decode and extract just the new generated text
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        results.append({
            "prompt_id": f"prompt_{i}",
            "instruction": item.get("instruction", ""),
            "input": item.get("input", ""),
            "expected_output": item.get("output", ""), # The ground truth
            "generated_output": response_text
        })
        
    return results

def main():
    print("Initializing Phase 4 Mass Inference...")

    # Load configs
    with open("config/config.json", "r") as f:
        config = json.load(f)
    with open("config/prompts.json", "r") as f:
        prompts = json.load(f)

    MODEL_ID = config["models"]["student_model"]
    TEMPLATE = prompts["student_training_schema"]["format"]
    
    # Load Evaluation Datasets
    with open("data/eval/alpaca_eval.json", "r", encoding="utf-8") as f:
        alpaca_eval = json.load(f)
    with open("data/eval/json_eval.json", "r", encoding="utf-8") as f:
        json_eval = json.load(f)

    os.makedirs("logs/inference_results", exist_ok=True)

    # 1. Base Model Setup (4-bit)
    print(f"\n--- Loading Checkpoint 0: Untuned Base Model ({MODEL_ID}) ---")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" 
    
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )
    base_model.eval()

    # Generate Checkpoint 0
    print("Evaluating Checkpoint 0 on Alpaca Data...")
    cp0_alpaca = generate_responses(base_model, tokenizer, alpaca_eval, TEMPLATE)
    with open("logs/inference_results/cp0_alpaca_results.json", "w", encoding="utf-8") as f:
        json.dump(cp0_alpaca, f, indent=2)

    print("Evaluating Checkpoint 0 on JSON Data...")
    cp0_json = generate_responses(base_model, tokenizer, json_eval, TEMPLATE)
    with open("logs/inference_results/cp0_json_results.json", "w", encoding="utf-8") as f:
        json.dump(cp0_json, f, indent=2)

    # 2. Stage 1 Adapter Setup
    print("\n--- Loading Checkpoint 1: Stage 1 Alpaca Adapter ---")
    model_cp1 = PeftModel.from_pretrained(base_model, "models/stage1_adapter")
    model_cp1.eval()

    print("Evaluating Checkpoint 1 on Alpaca Data...")
    cp1_alpaca = generate_responses(model_cp1, tokenizer, alpaca_eval, TEMPLATE)
    with open("logs/inference_results/cp1_alpaca_results.json", "w", encoding="utf-8") as f:
        json.dump(cp1_alpaca, f, indent=2)

    print("Evaluating Checkpoint 1 on JSON Data...")
    cp1_json = generate_responses(model_cp1, tokenizer, json_eval, TEMPLATE)
    with open("logs/inference_results/cp1_json_results.json", "w", encoding="utf-8") as f:
        json.dump(cp1_json, f, indent=2)

    # Clear memory before loading next adapter
    del model_cp1
    gc.collect()
    torch.cuda.empty_cache()

    # 3. Stage 2 Adapter Setup
    print("\n--- Loading Checkpoint 2: Stage 2 JSON Instruct Adapter ---")
    model_cp2 = PeftModel.from_pretrained(base_model, "models/stage2_adapter")
    model_cp2.eval()

    print("Evaluating Checkpoint 2 on Alpaca Data...")
    cp2_alpaca = generate_responses(model_cp2, tokenizer, alpaca_eval, TEMPLATE)
    with open("logs/inference_results/cp2_alpaca_results.json", "w", encoding="utf-8") as f:
        json.dump(cp2_alpaca, f, indent=2)

    print("Evaluating Checkpoint 2 on JSON Data...")
    cp2_json = generate_responses(model_cp2, tokenizer, json_eval, TEMPLATE)
    with open("logs/inference_results/cp2_json_results.json", "w", encoding="utf-8") as f:
        json.dump(cp2_json, f, indent=2)

    print("\nMass Inference Complete! Results saved to logs/inference_results/")

if __name__ == "__main__":
    main()