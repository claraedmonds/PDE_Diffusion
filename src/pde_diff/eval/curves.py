"""Loss-curve plotting for a single (possibly k-fold) training run:
train/val loss per fold (combined into one plot), validation-only
per-residual curves scaled by that run's configured c_residual weights, and
rollout metrics vs lead time."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from omegaconf import DictConfig

from pde_diff.eval_primitives import PLOT_TYPE, VAR_UNITS, plot_training_metrics, plot_cv_individual_fold_curves, find_metrics_csv
from pde_diff.eval.residuals import get_residual_weights
from pde_diff.eval.rollout import RESIDUALS

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


# Display style per `member` value of the rollout metric rows.
MEMBER_STYLES = {
    "member_mean": dict(label="Forecast (member mean)", color="C0", ls="-", marker="o"),
    "ens_mean": dict(label="Ensemble mean", color="C1", ls="-", marker="s"),
    "ens": dict(label="Ensemble", color="C2", ls="-", marker="^"),
    "persistence": dict(label="Persistence", color="gray", ls="--", marker=None),
    "truth": dict(label="ERA5", color="black", ls="--", marker=None),
}


def lead_curve(df: pd.DataFrame, metric: str, member: str, variable: str = "all", level: str = "all"):
    """Mean and 95% CI half-width over inits of one rollout metric vs lead
    time, or None if no rows match."""
    sub = df[(df["metric"] == metric) & (df["member"] == member)
             & (df["variable"] == variable) & (df["level"] == level)]
    if sub.empty:
        return None
    g = sub.groupby("lead_h")["value"]
    mean = g.mean()
    ci = (1.96 * g.std(ddof=1) / np.sqrt(g.count())).fillna(0.0)
    return mean.index.values, mean.values, ci.values


def plot_lead_curves(df: pd.DataFrame, curves: list[tuple[str, str]], out_path: Path, variable: str = "all",
                     level: str = "all", ylabel: str | None = None, title: str | None = None) -> Path | None:
    """Plot `(metric, member)` curves vs lead time (mean +- 95% CI over
    inits) on one axis."""
    fig, ax = plt.subplots(figsize=(6, 4))
    any_plotted = False
    for metric, member in curves:
        curve = lead_curve(df, metric, member, variable, level)
        if curve is None:
            continue
        x, mean, ci = curve
        style = dict(MEMBER_STYLES.get(member, dict(label=member, color=None, ls="-", marker="o")))
        label = style.pop("label")
        if len({m for m, _ in curves}) > 1:
            label = f"{metric}: {label}"
        ax.plot(x, mean, label=label, **style)
        ax.fill_between(x, mean - ci, mean + ci, alpha=0.25, color=ax.lines[-1].get_color())
        any_plotted = True
    if not any_plotted:
        plt.close(fig)
        return None

    ax.set_xlabel("Lead time (h)")
    ax.set_ylabel(ylabel or curves[0][0])
    ax.set_title(title or f"{curves[0][0]} vs lead time ({variable}, {level})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def summarize_rollout(df: pd.DataFrame) -> pd.DataFrame:
    """Mean, 95% CI half-width and number of inits per
    (metric, member, variable, level, lead_h)."""
    g = df.groupby(["metric", "member", "variable", "level", "lead_h"])["value"]
    out = g.agg(mean="mean", std="std", n="count").reset_index()
    out["ci95"] = (1.96 * out["std"] / np.sqrt(out["n"])).fillna(0.0)
    return out.drop(columns="std")


def plot_rollout_summary(df: pd.DataFrame, out_dir: Path, var_names: list[str], level: str = "500") -> None:
    """Standard set of rollout plots: MSE (overall and per variable at
    `level`), R1-R3 residuals against the ERA5 reference, CRPS, spread vs
    skill, and lat-weighted RMSE per variable."""
    out_dir = Path(out_dir)
    plots = [
        (dict(curves=[("mse", "member_mean"), ("mse", "ens_mean"), ("mse", "persistence")],
              ylabel="MSE (normalised)"), "mse_all"),
        (dict(curves=[("crps", "ens")], ylabel="CRPS (normalised)"), "crps_all"),
        (dict(curves=[("spread", "ens"), ("ens_mean_rmse", "ens_mean")], ylabel="Normalised units",
              title="Ensemble spread vs ensemble-mean RMSE"), "spread_skill_all"),
    ]
    for name, _, _ in RESIDUALS:
        plots.append((dict(curves=[(name, "member_mean"), (name, "truth")], ylabel=f"mean |{name.split('_')[0]}| (normalised)",
                           title=name.replace("_", " ")), name))
    for var in var_names:
        plots.append((dict(curves=[("mse", "member_mean"), ("mse", "ens_mean"), ("mse", "persistence")],
                           variable=var, level=level, ylabel="MSE (normalised)"), f"mse_{var}{level}"))
        plots.append((dict(curves=[("rmse", "member_mean"), ("rmse", "ens_mean"), ("rmse", "persistence")],
                           variable=var, level=level, ylabel=f"RMSE ({VAR_UNITS.get(var, '')})"), f"rmse_{var}{level}"))

    for kwargs, name in plots:
        path = plot_lead_curves(df, out_path=out_dir / f"{name}{PLOT_TYPE}", **kwargs)
        if path is not None:
            print(f"Saved: {path}")
