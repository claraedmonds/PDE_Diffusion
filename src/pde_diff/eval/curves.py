"""Loss-curve plotting for a single (possibly k-fold) training run:
train/val loss per fold (combined into one plot), and validation-only
per-residual curves scaled by that run's configured c_residual weights."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from omegaconf import DictConfig

from pde_diff.eval_primitives import PLOT_TYPE, plot_training_metrics, plot_cv_individual_fold_curves, find_metrics_csv
from pde_diff.eval.residuals import get_residual_weights

# metrics.csv column -> (residual short name, index into c_residual list)
RESIDUAL_COLUMNS = [
    ("val_era5_planetary_residual(norm)", "planetary vorticity", 0),
    ("val_era5_geo_wind_residual(norm)", "geostrophic wind", 1),
    ("val_era5_vort_div_residual(norm)", "vorticity divergence", 2),
]


def plot_loss_curves(model_id: str, out_dir: Path, fold_num: int | None = None, log_path: str = "logs") -> None:
    """Plot train/val loss vs epoch. When `fold_num` is set, each fold's
    curve is drawn as a separate line on one shared-axes plot (not one plot
    per fold) instead of the unfolded single-run plot — a k-fold model has
    no `logs/<model_id>/` directory of its own, only `<model_id>-<fold>/`."""
    if fold_num:
        plot_cv_individual_fold_curves(model_id, fold_num, log_path=log_path, out_dir=out_dir.parent)
    else:
        plot_training_metrics(model_id, out_dir=out_dir.parent)


def _read_metrics(model_id: str, log_path: str = "logs") -> pd.DataFrame | None:
    try:
        csv_path = find_metrics_csv(model_id, log_path=log_path)
    except FileNotFoundError:
        return None
    return (
        pd.read_csv(csv_path)
        .apply(pd.to_numeric, errors="coerce")
        .dropna(subset=["step"])
        .sort_values("step")
    )


def plot_residual_curves(
    model_id: str,
    model_cfg: DictConfig,
    out_dir: Path,
    fold_num: int | None = None,
    log_path: str = "logs",
) -> None:
    """Validation-only per-residual-component loss curves, scaled by the
    model's configured `loss.c_residual` weights (no training-code changes:
    this plots whatever is already logged in metrics.csv for the val split).
    """
    weights = get_residual_weights(model_cfg)
    if not any(w != 0.0 for w in weights):
        print(f"  All residual weights are 0 for {model_id}; skipping residual-curve plot.")
        return

    model_ids = [model_id] if not fold_num else [f"{model_id}-{f}" for f in range(1, fold_num + 1)]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    any_plotted = False
    cmap = plt.get_cmap("tab10")

    for col, name, weight_idx in RESIDUAL_COLUMNS:
        weight = weights[weight_idx] if weight_idx < len(weights) else 0.0
        if weight == 0.0:
            continue

        per_fold_series = []
        epochs_ref = None
        for mid in model_ids:
            df = _read_metrics(mid, log_path=log_path)
            if df is None or col not in df.columns:
                continue
            sub = df[["epoch", col]].dropna()
            if sub.empty:
                continue
            per_fold_series.append(sub[col].values * weight)
            if epochs_ref is None:
                epochs_ref = sub["epoch"].values

        if not per_fold_series:
            continue

        min_len = min(len(s) for s in per_fold_series)
        stacked = np.stack([s[:min_len] for s in per_fold_series])
        mean = stacked.mean(axis=0)
        epochs = epochs_ref[:min_len]

        color = cmap(weight_idx)
        ax.plot(epochs, mean, label=f"{name} (c={weight:g})", color=color)
        if stacked.shape[0] > 1:
            std = stacked.std(axis=0)
            half = 1.96 * std / np.sqrt(stacked.shape[0])
            ax.fill_between(epochs, mean - half, mean + half, alpha=0.3, color=color)
        any_plotted = True

    if not any_plotted:
        plt.close(fig)
        print(f"  No residual columns found for {model_id}; skipping residual-curve plot.")
        return

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Scaled validation residual")
    ax.set_title(f"Validation per-residual loss — {model_id}")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, fancybox=True, framealpha=0.9)
    fig.tight_layout()

    save_dir = Path(out_dir) / model_id
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"val_residual_curves{PLOT_TYPE}"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_forecast_loss_vs_steps(df, figsize=(8, 5), dir=None, loss_name=None):
    """Reads a forecasting-losses CSV and plots mean loss vs forecast step
    with a 95% CI band."""
    cols = list(df.columns)
    try:
        x = sorted([int(c) for c in cols])
        cols_sorted = [str(c) for c in x]
    except Exception:
        cols_sorted = cols

    fig, ax = plt.subplots(figsize=figsize)
    mean = df[cols_sorted].astype(float).mean(axis=0)
    std = df[cols_sorted].astype(float).std(axis=0)
    confidence = 1.96 * std / np.sqrt(len(df))
    ax.plot(range(1, len(cols_sorted) + 1), mean.values, marker='o', label='Mean')
    ax.fill_between(range(1, len(cols_sorted) + 1), (mean - confidence).values, (mean + confidence).values, alpha=0.3, label='Mean 95% CI')
    ax.set_xlabel('Forecast step')
    ax.set_ylabel(f'{loss_name} Loss')
    ax.set_title('Forecast loss vs forecast steps')
    ax.set_xticks(range(1, len(cols_sorted) + 1))
    ax.legend()
    fig.tight_layout()

    dir = Path(dir) if dir is not None else Path(".")
    dir.mkdir(parents=True, exist_ok=True)
    save_path = dir / f'forecast_loss_vs_steps_{loss_name}.png'
    fig.savefig(save_path)
    plt.close(fig)
    print(f'Forecast loss vs steps plot saved to {save_path}.')
