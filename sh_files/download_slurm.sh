#!/bin/bash
#SBATCH --job-name=download_era5
#SBATCH --output=logs/download/era5_%j.out
#SBATCH --error=logs/download/era5_%j.err
#SBATCH --partition=defq
#SBATCH --account=iu_0104
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
# end of SBATCH options

mkdir -p logs/download

source /dcai/users/emdcla/miniconda3/etc/profile.d/conda.sh
conda activate pde_diff

python src/pde_diff/data/download_data.py --year $YEAR
