"""Multi-model comparison: overlay rollout-metric-vs-lead curves and train/val loss
curves across an arbitrary list of models, using a generated color palette
(no hardcoded model names/colors)."""
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import matplotlib.pyplot as plt
from omegaconf import DictConfig

from pde_diff.eval.loading import find_best_fold
from pde_diff.eval.palette import get_model_color
from pde_diff.eval.curves import _read_metrics, lead_curve, MEMBER_STYLES
from pde_diff.eval.rollout import RESIDUALS, load_rollout_metrics


# (metric, member, reference member drawn once in black) overlaid across models.
ROLLOUT_COMPARISONS = [
    ("mse", "member_mean", "persistence"),
    ("mse", "ens_mean", "persistence"),
    ("crps", "ens", None),
    ("spread", "ens", None),
    *[(name, "member_mean", "truth") for name, _, _ in RESIDUALS],
]


def plot_rollout_comparison(dfs_by_model: dict[str, pd.DataFrame], metric: str, member: str, out_dir: Path,
                            reference_member: str | None = None, variable: str = "all", level: str = "all") -> Optional[Path]:
    """Overlay mean+95%CI rollout-metric-vs-lead curves for an arbitrary
    number of models, plus an optional model-independent reference curve
    (ERA5 truth residuals, persistence) taken from the first model that has it."""
    fig, ax = plt.subplots(figsize=(5, 4))
    n_models = len(dfs_by_model)
    any_plotted = False

    for idx, (label, df) in enumerate(dfs_by_model.items()):
        curve = lead_curve(df, metric, member, variable, level)
        if curve is None:
            continue
        x, mean, ci = curve
        color = get_model_color(idx, n_models)
        ax.plot(x, mean, "o-", label=label, color=color)
        ax.fill_between(x, mean - ci, mean + ci, alpha=0.3, color=color)
        any_plotted = True

    if reference_member is not None:
        for df in dfs_by_model.values():
            curve = lead_curve(df, metric, reference_member, variable, level)
            if curve is not None:
                x, mean, _ = curve
                ax.plot(x, mean, "--", color="black", label=MEMBER_STYLES[reference_member]["label"])
                break

    if not any_plotted:
        plt.close(fig)
        return None

    ax.set_xlabel("Lead time (h)")
    ax.set_ylabel(f"{metric} ({member})")
    ax.set_title(f"{metric} vs lead time")
    ax.legend()
    fig.tight_layout()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"compare_rollout_{metric}_{member}.png"
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


def run_comparison(cfg: DictConfig, run_single_fn: Callable[[DictConfig, str], Path]) -> None:
    """Compare `cfg.eval.compare.model_ids` on rollout-metric-vs-lead curves
    and train/val loss curves.

    `run_single_fn(cfg, model_id) -> eval_output_dir` is injected (rather
    than imported) to avoid a circular import with eval.run, and is only
    invoked (in-process, no subprocess) when `cfg.eval.compare.rerun_missing`
    is true and a model's rollout metrics don't exist yet.
    """
    model_ids = list(cfg.eval.compare.model_ids)
    labels = dict(cfg.eval.compare.get("labels", {}))
    out_dir = Path(cfg.eval.output_dir) / "compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    resolved_model_ids = []
    dfs_by_model: dict[str, pd.DataFrame] = {}

    for model_id in model_ids:
        best_fold_id, _ = find_best_fold(f"models/{model_id}")
        display_name = labels.get(model_id, model_id)
        resolved_model_ids.append(best_fold_id)

        eval_dir = Path(cfg.eval.output_dir) / f"{best_fold_id}_eval"
        df = load_rollout_metrics(eval_dir)
        if df is None and cfg.eval.compare.get("rerun_missing", True):
            print(f"No rollout metrics found for {model_id}; running single-model eval first.")
            eval_dir = run_single_fn(cfg, model_id)
            df = load_rollout_metrics(eval_dir)
        if df is None:
            print(f"No rollout metrics for {model_id} in {eval_dir}; leaving it out of the rollout comparison.")
            continue
        dfs_by_model[display_name] = df

    if dfs_by_model:
        for metric, member, reference in ROLLOUT_COMPARISONS:
            plot_rollout_comparison(dfs_by_model, metric, member, out_dir, reference_member=reference)

    plot_loss_curve_comparison(resolved_model_ids, out_dir, metric="train_loss")
    plot_loss_curve_comparison(resolved_model_ids, out_dir, metric="val_loss")
