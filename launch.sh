#!/bin/bash
source /etc/profile.d/modules.sh

module load anaconda3
module load cuda/9.0
#module load cudnn/v6.0
#module load NCCL/2.0.5
source activate pytorch03
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
MASTER_ADDR="${SLURM_NODELIST//[}"
MASTER_ADDR="${MASTER_ADDR%-*}"
echo "RANK=$SLURM_PROCID"
echo "MASTER_ADDR=$MASTER_ADDR"
RANK="$SLURM_PROCID" MASTER_ADDR="$MASTER_ADDR" MASTER_PORT=29500 PYTHONPATH=. python examples/train_model.py -m jenga $@
