import json
import os
from rouge_score import rouge_scorer

def calculate_json_validity(file_path):
    """Calculates just the JSON validity percentage as a proxy for JSON capability."""
    if not os.path.exists(file_path):
        return 0.0
        
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    valid_count = 0
    for item in data:
        generated = item.get("generated_output", "").strip()
        for prefix in ["```json", "```"]:
            if generated.startswith(prefix):
                generated = generated[len(prefix):]
        if generated.endswith("```"):
            generated = generated[:-3]
        generated = generated.strip()
        
        try:
            json.loads(generated)
            valid_count += 1
        except Exception:
            pass
            
    return (valid_count / len(data)) * 100 if data else 0

def calculate_rouge_l(file_path):
    """Calculates ROUGE-L as a proxy for Alpaca capability/forgetting."""
    if not os.path.exists(file_path):
        return 0.0
        
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l_scores = []
    
    for item in data:
        scores = scorer.score(item.get("expected_output", ""), item.get("generated_output", ""))
        rouge_l_scores.append(scores['rougeL'].fmeasure)
        
    return (sum(rouge_l_scores) / len(rouge_l_scores)) * 100 if rouge_l_scores else 0

def main():
    print("Calculating Epoch Ablation Metrics...\n")
    
    # 1. JSON Validity across epochs (Did it learn JSON fast?)
    json_1ep = calculate_json_validity("data/eval/inference_results/cp2_1ep_json_results.json")
    json_2ep = calculate_json_validity("data/eval/inference_results/cp2_2ep_json_results.json")
    json_3ep = calculate_json_validity("data/eval/inference_results/cp2_json_results.json") # The main CP2 run
    
    # 2. Alpaca ROUGE-L across epochs (Did it forget Alpaca slowly or instantly?)
    alp_1ep = calculate_rouge_l("data/eval/inference_results/cp2_1ep_alpaca_results.json")
    alp_2ep = calculate_rouge_l("data/eval/inference_results/cp2_2ep_alpaca_results.json")
    alp_3ep = calculate_rouge_l("data/eval/inference_results/cp2_alpaca_results.json") # The main CP2 run

    # Print the table (Section 4.5)
    print("### 4.5 Ablation Study: Stage 2 Training Epochs")
    print("| Stage 2 Epochs | JSON Validity (Capability Gained) | Alpaca ROUGE-L (Capability Retained) |")
    print("| :--- | :--- | :--- |")
    print(f"| 1 Epoch | {json_1ep:.1f}% | {alp_1ep:.1f} |")
    print(f"| 2 Epochs | {json_2ep:.1f}% | {alp_2ep:.1f} |")
    print(f"| 3 Epochs (Main) | {json_3ep:.1f}% | {alp_3ep:.1f} |")

    print("\nAnalysis Tip:")
    print("If JSON Validity is already high at 1 Epoch, but ROUGE-L drops heavily by 3 Epochs,")
    print("you can conclude that 1 Epoch is the optimal hyperparameter to prevent catastrophic forgetting!")

if __name__ == "__main__":
    main()