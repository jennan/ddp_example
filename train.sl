#!/bin/bash -e
#SBATCH --job-name=ddp_train
#SBATCH --partition=hgx
#SBATCH --time=00-00:05:00
#SBATCH --gpus-per-node=A100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# load modules
module purge
module load CUDA/11.6.2
module load Miniconda3/22.11.1-1
source $(conda info --base)/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1

# display information about the available GPUs
nvidia-smi

# check the value of the CUDA_VISIBLE_DEVICES variable
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# activate conda environment
conda deactivate
conda activate ./venv
which python

# start training script
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \  # TODO replace with number from SLURM
    train.py
