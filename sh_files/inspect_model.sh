#!/bin/bash
#SBATCH --job-name=inspect_model
#SBATCH --output=logs/inspect_model/%j.out
#SBATCH --error=logs/inspect_model/%j.err
#SBATCH --partition=defq
#SBATCH --account=iu_0104
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
# end of SBATCH options

# Usage: sbatch sh_files/inspect_model.sh <model-id> [n-samples]
#   e.g. sbatch sh_files/inspect_model.sh tiny-first_unet3d_conditional

MODEL_ID=${1:?"Usage: sbatch inspect_model.sh <model-id> [n-samples]"}
N_SAMPLES=${2:-3}

mkdir -p logs/inspect_model

source /dcai/users/emdcla/miniconda3/etc/profile.d/conda.sh
conda activate pde_diff

python src/pde_diff/inspect_model.py \
    --model-id "${MODEL_ID}" \
    --n-samples "${N_SAMPLES}" \
    --out-dir reports/inspect
