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

# Full eval pipeline (test metrics, sample grids, PSD, full rollout) on the held-out years
# set in configs/evaluate.yaml (2023-2024). Writes to reports/eval/<name>_eval/.
#
# Usage:
#   Single model (one job per model, run in parallel):
#     sbatch sh_files/evaluate.sh eval.model.path=models/era5-downsampled3-1
#     sbatch sh_files/evaluate.sh eval.model.path=models/era5-r1c1e-2-1
#     sbatch sh_files/evaluate.sh eval.model.path=models/era5-new_divloss-1
#   Compare (once the single runs are done; finds their rollout metrics and only plots,
#   any model without metrics is rerun in-process):
#     sbatch sh_files/evaluate.sh eval.mode=compare \
#         'eval.compare.model_ids=[era5-downsampled3-1,era5-r1c1e-2-1,era5-new_divloss-1]'
#   PSD only, for a model whose rollout is already done (mean over eval.psd.n_inits inits x all
#   members at eval.psd.leads; compare mode then overlays every model's saved psd_rollout.npz):
#     sbatch sh_files/evaluate.sh eval.model.path=models/era5-r1c1e-2-1 eval.plots.rollout=false
#
# A path with a fold suffix (models/<id>-3) evaluates that fold. A path without one
# (models/<id>) picks the fold with the lowest validation MSE, which can differ per model.
# Re-submitting resumes the rollout, skipping inits already written.

if [[ " $* " != *" eval.model.path="* && " $* " != *" eval.mode=compare "* ]]; then
    echo "Pass a model or compare mode, e.g.: sbatch sh_files/evaluate.sh eval.model.path=models/<model-id>" >&2
    exit 1
fi

mkdir -p logs/evaluate

source /dcai/users/emdcla/miniconda3/etc/profile.d/conda.sh
conda activate pde_diff

python -m pde_diff.eval.run "$@"
