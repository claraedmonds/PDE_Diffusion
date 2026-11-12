#!/bin/bash
#SBATCH --job-name=era5_trajectory
#SBATCH --output=logs/precompute/%A.out
#SBATCH --error=logs/precompute/%A.err
#SBATCH --partition=defq
#SBATCH --account=iu_0104
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
# end of SBATCH options

# Builds the hourly test trajectory (2023-2024) used for rollout evaluation.
# Disk space required: ~5 GB (states + clock features).

mkdir -p logs/precompute

source /dcai/users/emdcla/miniconda3/etc/profile.d/conda.sh
conda activate pde_diff

python src/pde_diff/data/precompute_trajectory.py \
    --dataset-config configs/dataset/era5.yaml \
    --zarr-dir ./data/era5 \
    --years 2023 2024 \
    --stats-from ./data/era5/precomputed/metadata.json \
    --out-dir ./data/era5/test_trajectory
