#!/bin/bash
#SBATCH --job-name=stage1_alpaca
#SBATCH --partition=gpu1a100
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:1           
#SBATCH --time=12:00:00        
#SBATCH --output=logs/stage1_%j.out 

# 1. Load the Anaconda module
module load anaconda3

# 2. Activate environment
conda activate /work/UTSA_ID/llm_assignment3_env

# 3. Set Hugging Face cache
export HF_HOME="/work/UTSA_ID/.HF_cache"

# 4. Navigate to the project root directory
cd $SLURM_SUBMIT_DIR

# 5. Run the script
python scripts/training/train_stage1_alpaca.py
