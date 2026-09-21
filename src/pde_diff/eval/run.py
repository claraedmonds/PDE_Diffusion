"""Single Hydra-driven entry point for post-training inference/evaluation.

Usage (see sh_files/evaluate.sh):
    python -m pde_diff.eval.run eval.model.path=models/<model-id>
    python -m pde_diff.eval.run eval.mode=compare eval.compare.model_ids=[a,b]
"""
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
import lightning as pl
from omegaconf import DictConfig, OmegaConf, open_dict

import pde_diff.loss  # registers loss classes

from pde_diff.eval.loading import find_best_fold, load_model_config, load_model, build_test_dataloader, get_grid_lat
from pde_diff.eval.curves import plot_loss_curves, plot_residual_curves, plot_forecast_loss_vs_steps
from pde_diff.eval.rollout import evaluate_forecasting
from pde_diff.eval.samples import plot_sample_target_absdiff_stacked, generate_prediction_error_distributions
from pde_diff.eval.psd import plot_all_psd_variables
from pde_diff.eval import compare as compare_module

VARS = ["u", "v", "pv", "t", "z"]


def _resolve_eval_dataset(cfg: DictConfig) -> tuple[dict, str]:
    """Build eval-time `dataset.*` overrides + output-dir name suffix from
    `cfg.eval.dataset`.

    `era5_precomputed` models hold train and held-out data in the same
    `inputs.npy`/`targets.npy` blob (see metadata.json's n_per_year), so a
    held-out split is a later `min_year`/`max_year` range rather than a
    separate directory — `PrecomputedERA5Dataset` already slices by year
    internally. `eval.dataset.min_year`/`max_year` must be set explicitly.
    """
    dataset_type = cfg.eval.dataset.type
    name_suffix = "_eval" if dataset_type == "test" else f"_eval_{dataset_type}"
    if cfg.eval.steps != 5:
        name_suffix += f"_{cfg.eval.steps}steps"

    min_year = cfg.eval.dataset.get("min_year", None)
    max_year = cfg.eval.dataset.get("max_year", None)
    if min_year is None and max_year is None:
        raise ValueError(
            "eval.dataset.min_year / max_year were not set. Set them to the held-out "
            "year range to slice out of the precomputed inputs.npy/targets.npy file "
            "(e.g. eval.dataset.min_year=2023 eval.dataset.max_year=2023)."
        )

    overrides = {}
    if min_year is not None:
        overrides["min_year"] = min_year
    if max_year is not None:
        overrides["max_year"] = max_year
    return overrides, name_suffix


def run_single(cfg: DictConfig, model_id: str | None = None) -> Path:
    """Run the single-model evaluation pipeline; returns the output directory."""
    model_path = cfg.eval.model.path if model_id is None else f"models/{model_id}"
    base_id = Path(model_path).name
    models_root = Path(model_path).parent

    best_fold_id, best_val = find_best_fold(str(model_path), log_path="logs")
    print(f"Best fold: {best_fold_id} with val loss {best_val}")
    resolved_model_path = str(models_root / best_fold_id)

    model_cfg = load_model_config(resolved_model_path, cfg)

    dataset_overrides, name_suffix = _resolve_eval_dataset(cfg)
    OmegaConf.set_struct(model_cfg, True)
    with open_dict(model_cfg):
        for key, val in dataset_overrides.items():
            model_cfg.dataset[key] = val
        model_cfg.dataset.forecast_steps = cfg.eval.steps

    name = f"{model_cfg.experiment.name}-{model_cfg.id}{name_suffix}"
    eval_dir = Path(cfg.eval.output_dir) / name
    eval_dir.mkdir(parents=True, exist_ok=True)

    # --- Loss curves (train/val per fold, val-only per-residual) ----------
    # Plotted before loading the model checkpoint: these only need logs/ and
    # the saved config, so they still work even if the model itself fails to
    # load (e.g. a stale/unregistered model.name in an old run's config).
    if cfg.eval.plots.loss_curves:
        print("Plotting loss curves ...")
        plot_loss_curves(base_id, eval_dir, fold_num=cfg.eval.model.fold_num)
    if cfg.eval.plots.residual_curves:
        print("Plotting per-residual validation curves ...")
        plot_residual_curves(base_id, model_cfg, eval_dir, fold_num=cfg.eval.model.fold_num)

    pl.seed_everything(model_cfg.get("seed", 0))
    model = load_model(model_cfg, resolved_model_path)

    # --- Base test-set metrics (trainer.test) ------------------------------
    dataset_test, dataloader_test = build_test_dataloader(model_cfg, batch_size=model_cfg.experiment.hyperparameters.batch_size)
    logger = pl.pytorch.loggers.CSVLogger(str(cfg.eval.output_dir), name=name)
    trainer = pl.Trainer(accelerator="cuda" if torch.cuda.is_available() else "cpu", logger=logger)
    results = trainer.test(model, dataloader_test)
    print(f"Test results: {results}")
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    # --- Optional single-step forward-diffusion debug samples --------------
    if cfg.eval.plots.save_forward_debug_samples:
        conditionals, target_changes = next(iter(dataloader_test))
        conditionals = conditionals.to(model.device)
        target_changes = target_changes.to(model.device)
        with torch.no_grad():
            pred_changes = model.sample_loop(batch_size=1, conditionals=conditionals[None, 0])[0]
        dir_sample = eval_dir / "sample_0_forward"
        level_idx = 1  # 500 hPa
        for j, var in enumerate(VARS):
            ch = j * 3 + level_idx
            plot_sample_target_absdiff_stacked(pred_changes[ch], target_changes[0, ch], variable=var, sample_idx=0, dir=dir_sample)

    # --- Qualitative sample + PSD plots -------------------------------------
    if cfg.eval.plots.sample_grids or cfg.eval.plots.psd:
        cond, target = dataset_test[0]
        cond = cond.unsqueeze(0).to(model.device)
        target = target.to(model.device)
        with torch.no_grad():
            pred = model.sample_loop(batch_size=1, conditionals=cond)[0]

        if cfg.eval.plots.sample_grids:
            level_idx = 1  # 500 hPa
            for j, var in enumerate(VARS):
                ch = j * 3 + level_idx
                plot_sample_target_absdiff_stacked(pred[ch], target[ch], variable=var, sample_idx=0, dir=eval_dir / "sample_0")

        ## TODO: additionally, the PSD is only computed for 1 sample
        if cfg.eval.plots.psd:
            diff_means = torch.as_tensor(dataset_test.diff_means, device=model.device)
            diff_stds = torch.as_tensor(dataset_test.diff_stds, device=model.device)
            pred_un = (pred * diff_stds.view(-1, 1, 1) + diff_means.view(-1, 1, 1)).cpu().numpy()
            target_un = (target * diff_stds.view(-1, 1, 1) + diff_means.view(-1, 1, 1)).cpu().numpy()
            level_idx = 1
            gt_500 = np.stack([target_un[j * 3 + level_idx] for j in range(len(VARS))])
            pred_500 = np.stack([pred_un[j * 3 + level_idx] for j in range(len(VARS))])
            grid_lat = get_grid_lat(dataset_test)
            psd_path = plot_all_psd_variables(
                ground_truth_state=gt_500,
                predictions_by_model={base_id: pred_500},
                grid_lat=grid_lat,
                var_names=VARS,
                out_dir=eval_dir,
            )
            print(f"  Saved {psd_path}")

    # --- Autoregressive rollout ---------------------------------------------
    # TODO: "era5_test" (ERA5DatasetTest in data/datasets.py) is a subclass
    # of the zarr-backed ERA5Dataset only — there is no multi-step-ahead
    # equivalent for PrecomputedERA5Dataset. Rollout will fail for
    # era5_precomputed models (our current default) until a
    # PrecomputedERA5DatasetTest class is written that returns
    # forecast_steps-ahead targets from the same inputs.npy/targets.npy
    # blob, mirroring how ERA5DatasetTest extends ERA5Dataset. Left as-is
    # for now (set eval.plots.rollout=false to skip it in the meantime).

    if cfg.eval.plots.rollout:
        print("Evaluating forecasting performance...")
        dataset_test_multistep, _ = build_test_dataloader(model_cfg, dataset_name="era5_test")
        forecasting_losses = evaluate_forecasting(
            model,
            dataset_test_multistep,
            steps=cfg.eval.steps,
            dir=eval_dir,
            n_qualitative_samples=cfg.eval.n_qualitative_samples,
            plot_sample_grids=cfg.eval.plots.sample_grids,
        )
        for loss_name, loss_dict in forecasting_losses.items():
            df = pd.DataFrame(loss_dict)
            df.index = [f"Forecast{i+1}" for i in range(len(df))]
            df.index.name = "forecast_step"
            plot_forecast_loss_vs_steps(df, dir=eval_dir, loss_name=loss_name)
            path_csv = eval_dir / f"forecasting_losses_{loss_name}.csv"
            df.to_csv(path_csv)
            print(f"Forecasting losses saved to CSV {path_csv}.")

        for loss_name, loss_values in forecasting_losses.items():
            for step, values in loss_values.items():
                if len(values) == 0:
                    continue
                print(f"Step {step}: {loss_name} = {sum(values) / len(values)}")

        if cfg.eval.plots.error_distributions:
            print("Generating single-step forecast error distributions (step 1)...")
            out_dir_dist = eval_dir / "forecast_error_distributions"
            generate_prediction_error_distributions(model, dataset_test_multistep, max_hist_samples_per_step=10, out_dir=out_dir_dist)
            print(f"Saved forecast error distributions to {out_dir_dist}")

    print(f"\nAll outputs saved to {eval_dir}/")
    return eval_dir


@hydra.main(version_base=None, config_name="evaluate.yaml", config_path="../../../configs")
def main(cfg: DictConfig) -> None:
    if cfg.eval.mode == "single":
        run_single(cfg)
    elif cfg.eval.mode == "compare":
        compare_module.run_comparison(cfg, run_single_fn=run_single)
    else:
        raise ValueError(f"Unknown eval.mode: {cfg.eval.mode!r} (expected 'single' or 'compare')")


if __name__ == "__main__":
    main()
