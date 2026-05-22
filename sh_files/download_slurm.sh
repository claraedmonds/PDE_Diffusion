#!/bin/bash
#SBATCH --job-name=download_era5
#SBATCH --output=logs/download/era5_%j.out
#SBATCH --error=logs/download/era5_%j.err
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G                # ERA5 grib + zarr conversion is memory-heavy
#SBATCH --time=24:00:00          # downloading 11 years can take many hours
# end of SBATCH options

mkdir -p logs/download

# activate the virtual environment
source /dcai/users/emdcla/miniconda3/etc/profile.d/conda.sh
conda activate pde_diff

python src/pde_diff/data/download_data.py
