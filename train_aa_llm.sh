#!/bin/bash
#SBATCH --job-name=AA-CLIP-VisA-LLM
#SBATCH --partition=A100-80GB,A100-PCI,A100-40GB,L40S,RTXA6000
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=logs/aa_train_logs8/%x-%j.log

#THIS IS BATCH FILE TO TRAIN LLM integrated AA-CLIP

set -euo pipefail

# === Turn the LLM on for this run ===
export AA_USE_LLM=1

export AA_LLM_MODEL="deepseek-ai/deepseek-llm-7b-chat"
export AA_LLM_DEVICE=cpu


export HF_HOME=/netscratch/$USER/huggingface

mkdir -p logs/aa_train_logs8
mkdir -p ckpt/visa_llm_on8


EXTRA_PIP='python -m pip install --no-cache-dir "transformers==4.41.2" "accelerate>=0.27" "safetensors>=0.4.1" sentencepiece tokenizers'

srun \
  --container-image=/netscratch/$USER/enroot/aa_requirements_updated_container.sqsh \
  --container-mounts=/netscratch/$USER:/netscratch/$USER,"$PWD":"$PWD",/home/$USER:/home/$USER \
  --container-workdir="$PWD" \
  --export="AA_USE_LLM,AA_LLM_MODEL,AA_LLM_DEVICE,HF_HOME,TRANSFORMERS_NO_FLASH_ATTENTION=1,NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
  bash -lc "${EXTRA_PIP}; \
            python -m pip uninstall -y flash-attn flash_attn flash-attn-2 >/dev/null 2>&1 || true; \
            python - <<'PY'
import glob, os, shutil
for p in glob.glob('/usr/local/lib/python3.10/dist-packages/flash_attn*'):
    try:
        print('Removing:', p)
        if os.path.isdir(p): shutil.rmtree(p, ignore_errors=True)
        else: os.remove(p)
    except Exception as e:
        print(' ignore:', e)
PY
            echo \"[env] AA_USE_LLM=\${AA_USE_LLM} AA_LLM_MODEL=\${AA_LLM_MODEL}\"; \
            python - <<'PY'
from forward_utils import REAL_NAMES
from runtime_llm import get_visual_defect_adjectives
print('[SMOKE] LLM:', get_visual_defect_adjectives(REAL_NAMES['MVTec']['carpet'], class_name='carpet', dataset='MVTec'))
PY
            python train.py --dataset VisA --save_path ./ckpt/visa_llm_on8 --training_mode full_shot"




