#!/bin/bash
#SBATCH --job-name=era5_merge
#SBATCH --output=logs/merge_zarrs/%A.out
#SBATCH --error=logs/merge_zarrs/%A.err
#SBATCH --partition=defq
#SBATCH --account=iu_0104
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
# end of SBATCH options

# WARNING: --out-path below points to data/era5/zarr, which will be OVERWRITTEN
# if a zarr already exists there 
# Rename or back it up first if you need to keep it.

mkdir -p logs/merge_zarrs

source /dcai/users/emdcla/miniconda3/etc/profile.d/conda.sh
conda activate pde_diff

python src/pde_diff/data/merge_zarrs.py \
    --data-dir ./data/era5 \
    --years 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 \
    --out-path ./data/era5/zarr
