#!/bin/bash
#SBATCH --job-name=swinir_lora
#SBATCH --output=logs/swinir_lora_%A_%a.out
#SBATCH --error=logs/swinir_lora_%A_%a.err
#SBATCH --gres=gpu:titanv # a100-40:1  # or h100-47:1
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --mail-user=yuletian@u.nus.edu
#SBATCH --mail-type=END,FAIL

echo "================================================="
echo "Running job on host: $(hostname)"
echo "Array Index: $SLURM_ARRAY_TASK_ID"
echo "================================================="

# Activate Python virtual environment
source /home/y/yuletian/adv_denoise_SwinIR/.venv/bin/activate

# Change to the directory from which the job was submitted
cd "${SLURM_SUBMIT_DIR}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run training script
python train_gtsrb_lora.py --lr 1e-3 --epochs 5 --batch_size 32
