import json
import os
import torch
import gc
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

def generate_responses(model, tokenizer, dataset, prompt_template):
    results = []
    for i, item in enumerate(tqdm(dataset, desc="Generating")):
        formatted_prompt = prompt_template.replace("{instruction}", item.get("instruction", "")) \
                                          .replace("{input}", item.get("input", "")) \
                                          .replace("{output}", "")
        
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        results.append({
            "prompt_id": f"prompt_{i}",
            "instruction": item.get("instruction", ""),
            "input": item.get("input", ""),
            "expected_output": item.get("output", ""), 
            "generated_output": response_text
        })
        
    return results

def main():
    print("Initializing Ablation Inference (1-Epoch and 2-Epoch Adapters)...")

    with open("config/config.json", "r") as f:
        config = json.load(f)
    with open("config/prompts.json", "r") as f:
        prompts = json.load(f)

    MODEL_ID = config["models"]["student_model"]
    TEMPLATE = prompts["student_training_schema"]["format"]
    
    with open("data/eval/alpaca_eval.json", "r", encoding="utf-8") as f:
        alpaca_eval = json.load(f)
    with open("data/eval/json_eval.json", "r", encoding="utf-8") as f:
        json_eval = json.load(f)

    os.makedirs("logs/inference_results", exist_ok=True)

    # Base Model Setup
    print(f"\n--- Loading Base Model ({MODEL_ID}) ---")
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
        device_map="auto"
    )
    base_model.eval()

    # 1-Epoch Adapter
    print("\n--- Loading Checkpoint: 1-Epoch Adapter ---")
    model_1ep = PeftModel.from_pretrained(base_model, "models/stage2_adapter_1ep")
    model_1ep.eval()

    print("Evaluating 1-Epoch on Alpaca Data...")
    cp2_1ep_alpaca = generate_responses(model_1ep, tokenizer, alpaca_eval, TEMPLATE)
    with open("logs/inference_results/cp2_1ep_alpaca_results.json", "w", encoding="utf-8") as f:
        json.dump(cp2_1ep_alpaca, f, indent=2)

    print("Evaluating 1-Epoch on JSON Data...")
    cp2_1ep_json = generate_responses(model_1ep, tokenizer, json_eval, TEMPLATE)
    with open("logs/inference_results/cp2_1ep_json_results.json", "w", encoding="utf-8") as f:
        json.dump(cp2_1ep_json, f, indent=2)

    del model_1ep
    gc.collect()
    torch.cuda.empty_cache()

    # 2-Epoch Adapter
    print("\n--- Loading Checkpoint: 2-Epoch Adapter ---")
    model_2ep = PeftModel.from_pretrained(base_model, "models/stage2_adapter_2ep")
    model_2ep.eval()

    print("Evaluating 2-Epoch on Alpaca Data...")
    cp2_2ep_alpaca = generate_responses(model_2ep, tokenizer, alpaca_eval, TEMPLATE)
    with open("logs/inference_results/cp2_2ep_alpaca_results.json", "w", encoding="utf-8") as f:
        json.dump(cp2_2ep_alpaca, f, indent=2)

    print("Evaluating 2-Epoch on JSON Data...")
    cp2_2ep_json = generate_responses(model_2ep, tokenizer, json_eval, TEMPLATE)
    with open("logs/inference_results/cp2_2ep_json_results.json", "w", encoding="utf-8") as f:
        json.dump(cp2_2ep_json, f, indent=2)

    print("\nAblation Inference Complete! Results saved.")

if __name__ == "__main__":
    main()