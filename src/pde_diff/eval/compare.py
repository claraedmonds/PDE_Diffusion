"""Multi-model comparison: overlay forecast-vs-step curves and train/val loss
curves across an arbitrary list of models, using a generated color palette
(no hardcoded model names/colors)."""
import glob
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from omegaconf import DictConfig

from pde_diff.eval.loading import find_best_fold
from pde_diff.eval.palette import get_model_color
from pde_diff.eval.curves import _read_metrics


def load_loss_from_csv(csv_path):
    df = pd.read_csv(csv_path, index_col=0)

    cols = list(df.columns)
    try:
        x = sorted([int(c) for c in cols])
        cols_sorted = [str(c) for c in x]
    except Exception:
        cols_sorted = cols

    mean = df[cols_sorted].astype(float).mean(axis=0)
    std = df[cols_sorted].astype(float).std(axis=0)
    confidence = 1.96 * std / np.sqrt(len(df))
    return cols_sorted, mean, confidence


def plot_forecast_step_curves(csv_by_model: dict[str, Path], loss_name: str, out_dir: Path) -> Path:
    """Overlay mean+95%CI forecast-loss-vs-step curves for an arbitrary
    number of models."""
    fig, ax = plt.subplots(figsize=(5, 4))
    model_ids = list(csv_by_model.keys())
    n_models = len(model_ids)

    cols_sorted = None
    for idx, model_id in enumerate(model_ids):
        cols_sorted, mean, confidence = load_loss_from_csv(csv_by_model[model_id])
        color = get_model_color(idx, n_models)
        ax.plot(range(1, len(cols_sorted) + 1), mean.values, 'o-', label=model_id, color=color)
        ax.fill_between(range(1, len(cols_sorted) + 1), (mean - confidence).values, (mean + confidence).values, alpha=0.3, color=color)

    ax.set_xlabel("Forecast horizon (steps)")
    ax.set_ylabel(loss_name)
    ax.set_title(f"{loss_name} vs forecast step")
    if cols_sorted is not None:
        ax.set_xticks(range(1, len(cols_sorted) + 1))
    ax.legend()
    fig.tight_layout()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"compare_forecast_loss_vs_steps_{loss_name}.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved comparison plot to {out_path}")
    return out_path


def plot_loss_curve_comparison(model_ids: list[str], out_dir: Path, log_path: str = "logs", metric: str = "val_loss") -> Optional[Path]:
    """Overlay a single loss metric (e.g. val_loss) across multiple models,
    one line per model (mean across folds if the model id has one)."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    any_plotted = False
    n_models = len(model_ids)

    for idx, model_id in enumerate(model_ids):
        df = _read_metrics(model_id, log_path=log_path)
        if df is None or metric not in df.columns:
            continue
        sub = df[["epoch", metric]].dropna()
        if sub.empty:
            continue
        color = get_model_color(idx, n_models)
        ax.plot(sub["epoch"], sub[metric], label=model_id, color=color)
        any_plotted = True

    if not any_plotted:
        plt.close(fig)
        print(f"No '{metric}' data found for any of {model_ids}; skipping comparison plot.")
        return None

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} comparison")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"compare_{metric}{'_'.join(model_ids)[:60]}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison plot to {out_path}")
    return out_path


def _find_forecast_csvs(eval_dir: Path) -> list[Path]:
    return [Path(p) for p in glob.glob(str(Path(eval_dir) / "forecasting_losses_*.csv"))]


def run_comparison(cfg: DictConfig, run_single_fn: Callable[[DictConfig, str], Path]) -> None:
    """Compare `cfg.eval.compare.model_ids` on forecast-vs-step curves and
    train/val loss curves.

    `run_single_fn(cfg, model_id) -> eval_output_dir` is injected (rather
    than imported) to avoid a circular import with eval.run, and is only
    invoked (in-process, no subprocess) when `cfg.eval.compare.rerun_missing`
    is true and a model's forecasting CSVs don't exist yet.
    """
    model_ids = list(cfg.eval.compare.model_ids)
    labels = dict(cfg.eval.compare.get("labels", {}))
    out_dir = Path(cfg.eval.output_dir) / "compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    resolved_model_ids = []
    csvs_by_model_and_loss: dict[str, dict[str, Path]] = {}

    for model_id in model_ids:
        best_fold_id, _ = find_best_fold(f"models/{model_id}")
        display_name = labels.get(model_id, model_id)
        resolved_model_ids.append(best_fold_id)

        eval_dir = Path(cfg.eval.output_dir) / f"{best_fold_id}_eval"
        csvs = _find_forecast_csvs(eval_dir)
        if not csvs and cfg.eval.compare.get("rerun_missing", True):
            print(f"No forecasting-loss CSVs found for {model_id}; running single-model eval first.")
            eval_dir = run_single_fn(cfg, model_id)
            csvs = _find_forecast_csvs(eval_dir)

        for csv_path in csvs:
            loss_name = csv_path.stem.replace("forecasting_losses_", "")
            csvs_by_model_and_loss.setdefault(loss_name, {})[display_name] = csv_path

    for loss_name, csv_by_model in csvs_by_model_and_loss.items():
        if len(csv_by_model) < 1:
            continue
        plot_forecast_step_curves(csv_by_model, loss_name, out_dir)

    plot_loss_curve_comparison(resolved_model_ids, out_dir, metric="train_loss")
    plot_loss_curve_comparison(resolved_model_ids, out_dir, metric="val_loss")
