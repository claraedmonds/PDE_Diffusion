#!/bin/bash
#SBATCH --job-name=evaluate
#SBATCH --output=logs/evaluate/%j.out
#SBATCH --error=logs/evaluate/%j.err
#SBATCH --partition=defq
#SBATCH --account=iu_0104
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
# end of SBATCH options

# Usage:
#   Single model: sbatch sh_files/evaluate.sh eval.model.path=models/<model-id> [extra hydra overrides]
# Example: sbatch sh_files/evaluate.sh eval.model.path=models/era5-new_divloss eval.model.fold_num=3
#   Compare mode: sbatch sh_files/evaluate.sh eval.mode=compare eval.compare.model_ids=[baseline,c1e2_pv]
#   Quick sample-only inspection (old inspect_model.sh use case):
#     sbatch sh_files/evaluate.sh eval.model.path=models/<model-id> eval.plots.rollout=false

mkdir -p logs/evaluate

source /dcai/users/emdcla/miniconda3/etc/profile.d/conda.sh
conda activate pde_diff

python -m pde_diff.eval.run "$@"
