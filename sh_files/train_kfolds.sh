#!/bin/bash
#SBATCH --job-name=train_kfolds
#SBATCH --output=logs/train_kfolds/%A_%a.out
#SBATCH --error=logs/train_kfolds/%A_%a.err
#SBATCH --partition=defq
#SBATCH --account=iu_0104
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=1-3
# end of SBATCH options

# Usage: sbatch sh_files/train_kfolds.sh <run-id>
#   e.g. sbatch sh_files/train_kfolds.sh baseline

RUN_ID=${1:?"Usage: sbatch train_kfolds.sh <run-id>"}

mkdir -p logs/train_kfolds

source /dcai/users/emdcla/miniconda3/etc/profile.d/conda.sh
conda activate pde_diff

python src/pde_diff/train.py \
    experiment=era5 \
    dataset=era5_precomputed \
    dataset.min_year=2015 \
    dataset.max_year=2022 \
    loss.name=vorticity \
    loss.c_residual=0 \
    model.name=unet3d_conditional \
    model.save_best_model=True \
    wandb=False \
    k_folds=3 \
    id=${RUN_ID} \
    idx_fold=${SLURM_ARRAY_TASK_ID} \
