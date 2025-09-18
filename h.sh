#!/bin/bash
#SBATCH --job-name="AA-CLIP_Train"
#SBATCH --partition=A100-80GB,A100-PCI,A100-40GB,RTX3090
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/aaclip_train_%j.out
#SBATCH --error=logs/aaclip_train_%j.err

# Helpful runtime flags
export CUDA_VISIBLE_DEVICES=0
export NCCL_DEBUG=WARN
export TORCH_DISTRIBUTED_DEBUG=OFF
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# Launch AA-CLIP inside NVIDIA PyTorch container
srun \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_24.06-py3.sqsh \
  --container-workdir="$(pwd)" \
  --container-mounts=/netscratch/uppal:/netscratch/uppal,"$(pwd)":"$(pwd)" \
  --gpus=1 \
  --mem=64G \
  bash -c '
    echo "=== AA-CLIP training in container ==="

    # 1) Upgrade pip quietly
    python -m pip install --upgrade --quiet pip

    # 2) Freeze NumPy at 1.26.4 **before** anything links against it
    pip uninstall -y numpy || true
    pip install --no-cache-dir numpy==1.26.4

    # 3) Force-install OpenCV that matches this NumPy
    pip uninstall -y opencv-python opencv-python-headless || true
    pip install --no-cache-dir opencv-python-headless==4.9.0.80

    # 4) Install project deps *without* letting pip pull a newer NumPy / OpenCV
    pip install --no-cache-dir --no-deps -r requirements.txt ipdb

    # 5) Run AA-CLIP
    python train.py --shot 5 --save_path ./ckpt/mvtec --dataset mvtec
  '
