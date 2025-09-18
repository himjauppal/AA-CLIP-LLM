#!/bin/bash
#SBATCH --job-name=AA-CLIP-Test
#SBATCH --partition=A100-80GB,A100-PCI,L40S,RTXA6000         
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=80G
#SBATCH --time=07:00:00
#SBATCH --output=logs/aa_test_log4/%x-%j.log


#TESTING THE ORIGINAL AA-CLIP

set -euo pipefail

mkdir -p logs/aa_test_logs4

echo "[env] host=$(hostname) user=$USER"
echo "[env] pwd=$(pwd)"

export AA_USE_LLM=0

DATASET="${1:-MVTec}"
SAVE_PATH="/netscratch/uppal/AA-CLIP/ckpt/visa7withouterror_20250913_151800"  

echo "[args] dataset=${DATASET} save_path=${SAVE_PATH}"

srun \
  --container-image=/netscratch/uppal/enroot/aa_requirements_updated_container.sqsh \
  --container-mounts=/netscratch/$USER:/netscratch/$USER,"$PWD":"$PWD",/home/$USER:/home/$USER \
  --container-workdir="$PWD" \
  --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
  bash -lc 'python -m pip install --no-cache-dir "transformers==4.41.2" "safetensors>=0.4.1" >/dev/null 2>&1 || true; \
            python /netscratch/uppal/AA-CLIP/test1.py --save_path '"$SAVE_PATH"' --dataset MVTec'




