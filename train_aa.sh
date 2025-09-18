#!/bin/bash
#SBATCH --job-name=AA-CLIP
#SBATCH --partition=A100-80GB,A100-PCI,A100-40GB,L40S,RTXA6000
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=80G
#SBATCH --time=07:00:00
#SBATCH --output=logs/aa_train_logs7/%x-%j.log

#THIS IS BATCH FILE TO TRAIN BASE AA-CLIP

set -euo pipefail

# toggle LLM
export AA_USE_LLM=0
# export AA_LLM_MODEL=deepseek-ai/deepseek-llm-7b-chat
export PYTHONUNBUFFERED=1

mkdir -p logs/aa_train_logs7
RUN_ID=$(date +%Y%m%d_%H%M%S)         
SAVE=./ckpt/visa7withouterror_${RUN_ID}


srun \
  --container-image=/netscratch/uppal/enroot/aa_requirements_updated_container.sqsh \
  --container-mounts=/netscratch/$USER:/netscratch/$USER,"$PWD":"$PWD",/home/$USER:/home/$USER \
  --container-workdir="$PWD" \
  bash -lc 'python -m pip install --no-cache-dir "transformers==4.41.2" "safetensors>=0.4.1" && python train1.py --dataset VisA --save_path '"$SAVE"' --training_mode full_shot'





