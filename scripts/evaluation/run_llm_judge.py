import json
import os
import random
from openai import OpenAI
from tqdm import tqdm

# Environment setup for the VPN API (Set in terminal before running)
API_KEY = os.getenv("OPENAI_API_KEY", "dummy-key")
BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def clean_json_string(raw_response):
    """Strips markdown formatting to ensure strict JSON parsing."""
    raw_response = raw_response.strip()
    if raw_response.startswith("```json"):
        raw_response = raw_response[7:]
    elif raw_response.startswith("```"):
        raw_response = raw_response[3:]
    if raw_response.endswith("```"):
        raw_response = raw_response[:-3]
    return raw_response.strip()

def evaluate_pair(instruction, resp_a, resp_b, prompt_id, cp_a_name, cp_b_name, judge_prompts, judge_model):
    """Calls the Llama 70B Judge to evaluate two responses."""
    system_prompt = judge_prompts["system_prompt"]
    
    # Format the evaluation template
    eval_prompt = judge_prompts["prompt_template"] \
        .replace("{prompt_id}", prompt_id) \
        .replace("{instruction}", instruction) \
        .replace("{response_a}", resp_a) \
        .replace("{response_b}", resp_b) \
        .replace("{checkpoint_a}", cp_a_name) \
        .replace("{checkpoint_b}", cp_b_name)

    try:
        response = client.chat.completions.create(
            model=judge_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": eval_prompt}
            ],
            temperature=0.1, # Extremely low temperature for consistent judging
            max_tokens=1024
        )
        
        raw_output = response.choices[0].message.content
        cleaned_output = clean_json_string(raw_output)
        
        # Validate that the judge actually returned the required JSON schema
        json_obj = json.loads(cleaned_output)
        return json_obj
        
    except Exception as e:
        print(f"\nJudge failed on {prompt_id}: {e}")
        return None

def run_comparison(file_a, file_b, cp_a_name, cp_b_name, output_file, judge_prompts, judge_model):
    """Runs a full pairwise comparison across two result files."""
    print(f"\nComparing {cp_a_name} vs {cp_b_name}...")
    
    with open(file_a, "r", encoding="utf-8") as f:
        data_a = json.load(f)
    with open(file_b, "r", encoding="utf-8") as f:
        data_b = json.load(f)

    results = []
    
    # Iterate through paired outputs
    for i in tqdm(range(len(data_a)), desc=f"{cp_a_name} vs {cp_b_name}"):
        item_a = data_a[i]
        item_b = data_b[i]
        
        prompt_id = item_a["prompt_id"]
        instruction = item_a["instruction"]
        
        # Randomize order to prevent LLM positional bias (Judge models often favor Response A)
        swap = random.choice([True, False])
        
        if swap:
            resp_a, resp_b = item_b["generated_output"], item_a["generated_output"]
            real_cp_a, real_cp_b = cp_b_name, cp_a_name
        else:
            resp_a, resp_b = item_a["generated_output"], item_b["generated_output"]
            real_cp_a, real_cp_b = cp_a_name, cp_b_name

        judge_result = evaluate_pair(
            instruction=instruction,
            resp_a=resp_a,
            resp_b=resp_b,
            prompt_id=prompt_id,
            cp_a_name=real_cp_a,
            cp_b_name=real_cp_b,
            judge_prompts=judge_prompts,
            judge_model=judge_model
        )

        if judge_result:
            # Map the judge's A/B winner back to the actual checkpoints if we swapped them
            if swap and judge_result.get("winner") in ["A", "B"]:
                judge_result["winner"] = "B" if judge_result["winner"] == "A" else "A"
            
            results.append(judge_result)
            
        # Incremental save every 10 iterations
        if len(results) % 10 == 0:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

    # Final save
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Comparison complete! Saved to {output_file}")

def main():
    # Load configs
    with open("config/config.json", "r") as f:
        config = json.load(f)
    with open("config/prompts.json", "r") as f:
        prompts = json.load(f)

    JUDGE_MODEL = config["models"]["judge_model"]
    JUDGE_PROMPTS = prompts["judge_evaluation"]

    # We only run the LLM Judge on the Alpaca (general instruction) data to measure forgetting.
    # JSON data is evaluated via strict regex/parsing in a separate metrics script.
    
    # 1. Compare Checkpoint 0 vs Checkpoint 1 (Did Stage 1 work?)
    run_comparison(
        file_a="data/eval/inference_results/cp0_alpaca_results.json",  # Adjust path if needed
        file_b="data/eval/inference_results/cp1_alpaca_results.json",
        cp_a_name="CP0_Base",
        cp_b_name="CP1_Alpaca",
        output_file="logs/judge_cp0_vs_cp1.json",
        judge_prompts=JUDGE_PROMPTS,
        judge_model=JUDGE_MODEL
    )

    # 2. Compare Checkpoint 1 vs Checkpoint 2 (Did Stage 2 cause catastrophic forgetting?)
    run_comparison(
        file_a="data/eval/inference_results/cp1_alpaca_results.json",
        file_b="data/eval/inference_results/cp2_alpaca_results.json",
        cp_a_name="CP1_Alpaca",
        cp_b_name="CP2_JSON",
        output_file="logs/judge_cp1_vs_cp2.json",
        judge_prompts=JUDGE_PROMPTS,
        judge_model=JUDGE_MODEL
    )
    
    # 3. Compare Checkpoint 1 vs Checkpoint 2 on JSON Data (Qualitative JSON scoring)
    run_comparison(
        file_a="data/eval/inference_results/cp1_json_results.json",
        file_b="data/eval/inference_results/cp2_json_results.json",
        cp_a_name="CP1_Alpaca",
        cp_b_name="CP2_JSON",
        output_file="logs/judge_cp1_vs_cp2_json.json",
        judge_prompts=JUDGE_PROMPTS,
        judge_model=JUDGE_MODEL
    )

if __name__ == "__main__":
    main()