#!/bin/bash
NOW=$( date '+%F-%H-%M-%S' )
JOB_NAME="MCVD-train"
PARTITION="L40S"
NODES=1
NTASKS=1 
CPUS_PER_GPU=4
# GPUS_PER_TASK=2
MEM=80G
GPUS=1

export LOGLEVEL=INFO
srun -K\
    --job-name=$JOB_NAME \
    --partition=$PARTITION \
    --nodes=$NODES \
    --gpus=$GPUS \
    --ntasks=$NTASKS \
    --cpus-per-gpu=$CPUS_PER_GPU \
    --mem=$MEM \
    --container-image=/netscratch/uppal/enroot/aa_requirements_for_refiner.sqsh \
    --container-mounts=/netscratch/$USER:/netscratch/$USER,"`pwd`":"`pwd`",/home/$USER:/home/$USER \
    --container-workdir="`pwd`" \
    --time=01:00:00 \
    --output="./logs/aa_refiner/${NOW}-${JOB_NAME}-nodes_${NODES}-tasks_${NTASKS}-cpus_${CPUS_PER_TASK}-gpus_${GPUS_PER_TASK}.log" \
    --export="NCCL_SOCKET_IFNAME=bond, NCCL_1B_HCA=mI×5" \
    python /netscratch/uppal/AA-CLIP/text_refiner.py



