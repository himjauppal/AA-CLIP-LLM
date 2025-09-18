#!/bin/bash
#SBATCH --job-name=AA-CLIP-Test
#SBATCH --partition=A100-80GB,A100-PCI,A100-40GB,L40S,RTXA6000
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=80G
#SBATCH --time=07:00:00
#SBATCH --output=logs/aa_test_logs/visa2mvtec-llm-%j.log

#TESTING FOR LLM AA-CLIP

set -euo pipefail


echo "[env] host=$(hostname) user=$USER"
echo "[env] pwd=$(pwd)"



DATASET="${1:-MVTec}"
SAVE_PATH="./ckpt/visa_llm_on8"  

echo "[args] dataset=${DATASET} save_path=${SAVE_PATH}"


export AA_USE_LLM=1
export AA_LLM_MODEL=deepseek-ai/deepseek-llm-7b-chat

# debug check
echo "AA_USE_LLM=$AA_USE_LLM"
echo "AA_LLM_MODEL=$AA_LLM_MODEL"


srun \
  --container-image=/netscratch/uppal/enroot/aa_requirements_updated_container.sqsh \
  --container-mounts=/netscratch/$USER:/netscratch/$USER,"$PWD":"$PWD",/home/$USER:/home/$USER \
  --container-workdir="$PWD" \
  --export="AA_USE_LLM=1,AA_LLM_MODEL=deepseek-ai/deepseek-llm-7b-chat,NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
  bash -lc 'python -m pip install --no-cache-dir "transformers==4.41.2" "accelerate>=0.27" "safetensors>=0.4.1" sentencepiece >/dev/null 2>&1 || true
    python -m pip uninstall -y flash-attn flash_attn flash-attn-2 >/dev/null 2>&1 || true
    python - <<PY
import glob, os, shutil
for p in glob.glob("/usr/local/lib/python3.10/dist-packages/flash_attn*"):
    try:
        print("Removing:", p)
        if os.path.isdir(p): shutil.rmtree(p, ignore_errors=True)
        else: os.remove(p)
    except Exception as e:
        print(" ignore:", e)
PY
    python - <<PY
import importlib.util as u
print("flash_attn:", u.find_spec("flash_attn"))
print("flash_attn_2_cuda:", u.find_spec("flash_attn_2_cuda"))
PY

    # 3) Force PyTorch SDPA path inside your loader (no flash-attn)
    #    (Make sure runtime_llm.py uses attn_implementation="sdpa".)

    # 4) *Quick smoke test* (optional but useful; comment out if you want)
    python - <<PY
from forward_utils import REAL_NAMES
from runtime_llm import get_visual_defect_adjectives
print("[SMOKE] adjectives:", get_visual_defect_adjectives(REAL_NAMES["MVTec"]["carpet"], class_name="carpet", dataset="MVTec"))
PY

    # 5) Run evaluation
    python /netscratch/uppal/AA-CLIP/test.py --save_path '"$SAVE_PATH"' --dataset '"$DATASET"'
  '
