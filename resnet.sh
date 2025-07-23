#!/bin/bash
#SBATCH --job-name=resnet_train
#SBATCH --output=logs/resnet_train_%A_%a.out
#SBATCH --error=logs/resnet_train_%A_%a.err
#SBATCH --gres=gpu:a100-40:1 # or h100-47:1
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --mail-user=yuletian@u.nus.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-2%2   # 3 learning rates, max 2 concurrent jobs

echo "================================================="
echo "Running job on host: $(hostname)"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "================================================="

# Activate your virtualenv
source /home/y/yuletian/adv_denoise_SwinIR/.venv/bin/activate

# Load W&B API key (ensure this file exists with proper permissions)
source ~/wandb_key.sh

# Define learning rates to sweep
learning_rates=(0.0001 0.0005 0.001)

# Select learning rate based on array index
lr=${learning_rates[$SLURM_ARRAY_TASK_ID]}

echo "Using learning rate: $lr"

# Train with fixed batch size (e.g. 64)
python train_resnet.py --lr $lr --epochs 10 --batch_size 64