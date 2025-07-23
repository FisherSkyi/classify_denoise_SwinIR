#!/bin/bash
#SBATCH --job-name=resnet_train
#SBATCH --output=logs/resnet_train_%A_%a.out
#SBATCH --error=logs/resnet_train_%A_%a.err
#SBATCH --gres=gpu:a100-40:1  # or h100-47:1
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --mail-user=yuletian@u.nus.edu
#SBATCH --mail-type=END,FAIL

echo "================================================="
echo "Running job on host: $(hostname)"
echo "Array Index: $SLURM_ARRAY_TASK_ID"
echo "================================================="
# Activate your virtualenv (optional if already activated)
#source /.venv/bin/activate

# Set W&B API key securely (or run `wandb login` once)

python train_resnet.py --lr 0.001 --epochs 10 --batch_size 64