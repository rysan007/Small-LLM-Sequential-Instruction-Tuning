#!/bin/bash
#SBATCH --job-name=stage2_ablation
#SBATCH --partition=gpu1a100
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:1           
#SBATCH --time=12:00:00        
#SBATCH --output=logs/stage2_ablation_%j.out 

# 1. Load the Anaconda module
module load anaconda3

# 2. Activate your specific environment
conda activate /work/UTSA_ID/llm_assignment3_env

# 3. Set Hugging Face cache
export HF_HOME="/work/UTSA_ID/.HF_cache"

# 4. Navigate to the project root directory
cd $SLURM_SUBMIT_DIR

# 5. Run the ablation scripts sequentially
echo "Running 1-epoch ablation..."
python scripts/training/train_stage2_json_ablation.py --epochs 1

echo "Running 2-epoch ablation..."
python scripts/training/train_stage2_json_ablation.py --epochs 2

echo "Ablation jobs finished successfully."