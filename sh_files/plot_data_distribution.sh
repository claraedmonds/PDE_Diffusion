#!/bin/bash
#SBATCH --job-name=plot_data_distribution
#SBATCH --output=logs/plot_data_distribution/%A.out
#SBATCH --error=logs/plot_data_distribution/%A.err
#SBATCH --partition=defq
#SBATCH --account=iu_0104
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
# end of SBATCH options

mkdir -p logs/plot_data_distribution

source /dcai/users/emdcla/miniconda3/etc/profile.d/conda.sh
conda activate pde_diff

python src/pde_diff/data/plot_data_distribution.py \
    --dataset-config configs/dataset/era5_precomputed.yaml \
    --out-json src/pde_diff/data/residual_stats.json
