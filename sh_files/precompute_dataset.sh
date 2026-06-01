#!/bin/bash
#SBATCH --job-name=era5_precompute
#SBATCH --output=logs/precompute/%A.out
#SBATCH --error=logs/precompute/%A.err
#SBATCH --partition=defq
#SBATCH --account=iu_0104
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=04:00:00
# end of SBATCH options

# The script loads the full downsampled dataset (~20 GB) into RAM first,
# then iterates in pure numpy. Expected runtime: ~1-2h total.
# Disk space required: ~90 GB for the full 10-year dataset.

mkdir -p logs/precompute

source /dcai/users/emdcla/miniconda3/etc/profile.d/conda.sh
conda activate pde_diff

python src/pde_diff/data/precompute_dataset.py \
    --dataset-config configs/dataset/era5.yaml \
    --out-dir ./data/era5/precomputed
