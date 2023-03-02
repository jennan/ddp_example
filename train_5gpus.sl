#!/bin/bash -e
#SBATCH --partition=hgx
#SBATCH --time=00-00:05:00
#SBATCH --ntasks=5
#SBATCH --gpus-per-task=A100:1
#SBATCH --gpu-bind=single:1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-gpu=4GB
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.out

# load modules
module purge
module load CUDA/11.6.2
module load Miniconda3/22.11.1-1
source $(conda info --base)/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1

# activate conda environment
conda deactivate
conda activate ./venv

# optional, used to peek under NCCL's hood
export NCCL_DEBUG=INFO

# start training script
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

echo "rendez-vous at $MASTER_ADDR:$MASTER_PORT"

# TODO pass the number of available CPUs from Slurm
srun bash -c 'torchrun \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=static \
    --node_rank=$SLURM_PROCID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --nnodes=$SLURM_NTASKS \
    --nproc_per_node=1 \
    train.py'
