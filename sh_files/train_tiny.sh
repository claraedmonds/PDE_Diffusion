#!/bin/bash
#SBATCH --job-name=train_tiny
#SBATCH --output=logs/train_tiny/%j.out
#SBATCH --error=logs/train_tiny/%j.err
#SBATCH --partition=defq
#SBATCH --account=iu_0104
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
# end of SBATCH options

mkdir -p logs/train_tiny

source /dcai/users/emdcla/miniconda3/etc/profile.d/conda.sh
conda activate pde_diff

python src/pde_diff/train.py \
    experiment=tiny \
    dataset=era5_tiny \
    val_dataset=era5_val_tiny \
    loss.name=vorticity \
    loss.c_residual=0 \
    wandb=False \
    model.save_best_model=True
