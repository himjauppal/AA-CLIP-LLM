#!/bin/bash
NOW=$( date '+%F-%H-%M-%S' )
JOB_NAME="D_SAT"
PARTITION="L40S"
NODES=1
NTASKS=1
CPUS_PER_TASK=4
GPUS_PER_TASK=0
MEM=80G
GPUS=0
# NODES=2
# NTASKS=2
# CPUS_PER_TASK=2
# GPUS_PER_TASK=1
# MEM=30G
# CPUS_PER_TASK=1

# nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
# nodes_array=($nodes)
# head_node=${nodes_array[0]}
# head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
# Get the first node as the master node
# MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
# MASTER_PORT=29501

# # Set the world size (total number of processes)
# WORLD_SIZE=$(($NODES * $GPUS_PER_TASK))

# # Print the environment variables for debugging
# echo "MASTER_ADDR=$MASTER_ADDR"
# echo "MASTER_PORT=$MASTER_PORT"
# echo "WORLD_SIZE=$WORLD_SIZE"
# echo Node IP: $head_node_ip
export LOGLEVEL=INFO
srun \
    --job-name=$JOB_NAME \
    --partition=$PARTITION \
    --nodes=$NODES \
    --gpus=$GPUS \
    --ntasks=$NTASKS \
    --cpus-per-task=$CPUS_PER_TASK \
    --mem=$MEM \
    --container-image=/enroot/nvcr.io_nvidia_pytorch_24.06-py3.sqsh \
    --container-save=/netscratch/uppal/enroot/aa_requirements_for_refiner.sqsh \
    --container-mounts=/netscratch/$USER:/netscratch/$USER,"`pwd`":"`pwd`",/home/$USER:/home/$USER \
    --container-workdir="`pwd`" \
    --time=01:00:00 \
    --output="./logs/install/${NOW}-${JOB_NAME}-nodes_${NODES}-tasks_${NTASKS}-cpus_${CPUS_PER_TASK}-gpus_${GPUS_PER_TASK}.log" \
    --export="NCCL_SOCKET_IFNAME=bond, NCCL_1B_HCA=mI×5" \
    bash install_1.sh
    # torchrun --nnodes=$NODES --nproc_per_node=$GPUS_PER_TASK --master_port=2432 /home/kashyap/Master_thesis/DiT/train.py --model DiT-XL/2 --data-path /netscratch/kashyap/Temporal_diffusion/GEE_data/processed_data/Landsat_from_1984/classes

    # bash install.sh \

    # --task-prolog="`pwd`/gcloud_authentication.sh" \
    # --container-save=/netscratch/$USER/enroot/temporal_diffusion_dependencies.sqsh \
    # bash gcloud_authe`ntication.sh \
    # --container-save=/netscratch/$USER/enroot/diffusion_transformer_dependencies.sqsh \

# --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_22.07-py3.sqsh # System available images
    # --container-save=/netscratch/$USER/enroot/temporal_diffusion_dependencies.sqsh \
# --container-image=/netscratch/$USER/HAT/scripts/pip_dependencies.sqsh \
    # bash gcloud_authentication.sh \
#    --container-save=/netscratch/$USER/enroot/temporal_diffusion_dependencies.sqsh \

# To see available images:
# cd /netscratch/enroot/
# ls nvcr.io_nvidia_pytorch_* -l
    # bash gcloud_authentication.sh \

# Run in interractive mode to use debugging:
# --pty /bin/bash \

# https://slurm.schedmd.com/sbatch.html
