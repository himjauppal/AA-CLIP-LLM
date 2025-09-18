#!/bin/bash
# put your install commands here (remove lines you don't need):
# make sure you run `chmod a+x install.sh` in cluster to give permission to this batch file
# make sure only first task per node installs stuff, others wait

# DONEFILE="/tmp/install_done_${SLURM_JOBID}"

# if [[ $SLURM_LOCALID == 0 ]]; then
#   set -euo pipefail

#   echo "[install] apt + base deps"
#   apt update -y
#   apt clean

#   echo "[install] torch/cu121 + project deps"
#   pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
#   pip install -e ".[torch]"
#   pip install -r /netscratch/uppal/AA-CLIP/requirements.txt
#   pip install ipdb

#   echo "[install] remove flash-attn & apex everywhere"
#   pip uninstall -y flash-attn flash_attn flash-attn-2 apex || true
#   python - <<'PY'
# import glob, os, shutil
# for p in glob.glob('/usr/local/lib/python3.10/dist-packages/flash_attn*'):
#     print("Removing:", p)
#     if os.path.isdir(p):
#         shutil.rmtree(p, ignore_errors=True)
#     else:
#         try: os.remove(p)
#         except: pass
# for p in glob.glob('/usr/local/lib/python3.10/dist-packages/flash_attn*_cuda*'):
#     print("Removing:", p)
#     try: os.remove(p)
#     except: pass
# PY
#   python -c "import importlib.util as u; print('flash_attn spec:', u.find_spec('flash_attn'))"
#   python -c "import importlib.util as u; print('flash_attn_2_cuda spec:', u.find_spec('flash_attn_2_cuda'))"

#   echo "[install] HF stack for DeepSeek"
#   pip install --no-cache-dir "transformers==4.41.2" "accelerate>=0.27" "safetensors>=0.4.1" sentencepiece

#   echo "[install] pin OpenCV trio (your project uses this)"
#   pip uninstall -y opencv-python opencv-contrib-python opencv-contrib-python-headless || true
#   pip install opencv-python==4.5.5.64
#   pip install opencv-contrib-python==4.5.5.64
#   pip install opencv-python-headless==4.5.5.64

#   pip install openai

#   echo "[install] persist environment for all shells in container"
#   # If /etc/profile.d is read-only, change path to /root/.bashrc instead.
#   cat >/etc/profile.d/aa_env.sh <<'EOF'
# export HF_HOME=/netscratch/$USER/.cache/huggingface
# export TRANSFORMERS_CACHE=$HF_HOME/transformers
# export HF_HUB_ENABLE_HF_TRANSFER=1

# export AA_USE_LLM=1
# export AA_LLM_MODEL="deepseek-ai/deepseek-llm-7b-chat"
# EOF

#   echo "[install] pre-cache DeepSeek model to /netscratch (avoids re-downloads)"
#   python - <<'PY'
# import os
# from huggingface_hub import snapshot_download
# mid = os.environ.get("AA_LLM_MODEL", "deepseek-ai/deepseek-llm-7b-chat")
# cache = os.environ.get("HF_HOME", f"/netscratch/{os.environ.get('USER','root')}/.cache/huggingface")
# print("Caching to:", cache, "Model:", mid)
# snapshot_download(repo_id=mid, cache_dir=cache, local_files_only=False)
# print("Done caching.")
# PY

#   apt-get clean
#   touch "${DONEFILE}"
# else
#   # other local ranks wait until install is complete
#   while [[ ! -f "${DONEFILE}" ]]; do sleep 1; done
# fi




DONEFILE="/tmp/install_done_${SLURM_JOBID}"
if [[ $SLURM_LOCALID == 0 ]]; then  
  # put your install commands here (remove lines you don't need):
  apt update
  apt clean
  pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
  pip install -e ".[torch]"
  pip install -r /netscratch/uppal/AA-CLIP/requirements.txt
  pip install ipdb


  pip uninstall -y flash-attn flash_attn flash-attn-2 || true

python - <<'PY'
import glob, os, shutil
paths = glob.glob('/usr/local/lib/python3.10/dist-packages/flash_attn*')
for p in paths:
    print("Removing:", p)
    if os.path.isdir(p):
        shutil.rmtree(p, ignore_errors=True)
    elif os.path.isfile(p):
        os.remove(p)

PY
python -c "import importlib.util as u; print('flash_attn spec:', u.find_spec('flash_attn'))"
python -c "import importlib.util as u; print('flash_attn_2_cuda spec:', u.find_spec('flash_attn_2_cuda'))"
    
  

  pip uninstall -y apex || true


  # --- HuggingFace stack ---
  pip install --upgrade "transformers==4.41.2" "safetensors>=0.4.1" sentencepiece
  pip install --no-cache-dir "accelerate>=0.27"
  # pip install ffmpeg-python




  pip uninstall opencv-python
  pip uninstall opencv-contrib-python
  pip uninstall opencv-contrib-python-headless
  pip install opencv-python==4.5.5.64
  pip install opencv-contrib-python==4.5.5.64
  pip install opencv-python-headless==4.5.5.64

  pip install openai


  apt-get clean
  
  # Tell other tasks we are done installing
  touch "${DONEFILE}"
else
  # Wait until packages are installed
  while [[ ! -f "${DONEFILE}" ]]; do sleep 1; done
fi

