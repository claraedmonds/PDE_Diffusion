#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output=logs/eval/%j.out
#SBATCH --error=logs/eval/%j.err
#SBATCH --partition=defq
#SBATCH --account=iu_0104
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
# end of SBATCH options

# Usage: sbatch sh_files/eval.sh <model-id>
#   e.g. sbatch sh_files/eval.sh era5-downsampled3
#        sbatch sh_files/eval.sh era5-r1c1e-2

MODEL_ID=${1:?"Usage: sbatch eval.sh <model-id>"}

mkdir -p logs/eval

source /dcai/users/emdcla/miniconda3/etc/profile.d/conda.sh
conda activate pde_diff

python src/pde_diff/evaluate.py model.path=models/"${MODEL_ID}"
