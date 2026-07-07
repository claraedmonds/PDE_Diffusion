"""
Quick inspection script: loss curves + a few predicted vs target sample plots.

Usage:
    python src/pde_diff/inspect_model.py --model-id tiny-geqcd
    python src/pde_diff/inspect_model.py --model-id tiny-geqcd --n-samples 3 --out-dir reports/inspect
    python src/pde_diff/inspect_model.py --model-id era5_baseline-diagnostic-1 --fold-num 3

Reads:
    models/<model-id>/config.yaml          — training config
    models/<model-id>/best-val_loss-weights.pt — model weights (state dict)
    logs/<model-id>/version_0/metrics.csv  — training metrics (CSVLogger output)

Writes PNGs to <out-dir>/<model-id>/.
"""
import argparse
import re
from pathlib import Path

import einops
import numpy as np
import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf

from pde_diff.model import DiffusionModel
from pde_diff.utils import DatasetRegistry
import pde_diff.loss  # registers loss classes

# Reuse plotting helpers from visualize.py where possible.
# The module-level font/style setup in visualize.py may fail if times.ttf is
# missing; we guard the import and fall back to defaults.
try:
    from pde_diff.visualize import (
        visualize_era5_sample,
        plot_training_metrics,
        plot_cv_individual_fold_curves,
        VAR_NAMES,
        VAR_UNITS,
        COLOR_BARS,
        PLOT_TYPE,
    )
    _VISUALIZE_OK = True
except Exception:
    _VISUALIZE_OK = False
    PLOT_TYPE = ".png"
    VAR_NAMES = {"u": "u", "v": "v", "pv": "pv", "t": "T", "z": "Phi"}
    VAR_UNITS = {}
    COLOR_BARS = {}

VARS = ["u", "v", "pv", "t", "z"]
LEVELS = [450, 500, 550]


# ---------------------------------------------------------------------------
# Loss curves
# ---------------------------------------------------------------------------

def plot_loss_curves(model_id: str, out_dir: Path, fold_num: int | None = None) -> None:
    if _VISUALIZE_OK:
        # reuse the existing function which handles all metrics
        plot_training_metrics(model_id, out_dir=out_dir.parent)
        if fold_num:
            # model_id may be a specific fold (e.g. "era5_baseline-diagnostic-1");
            # strip the trailing "-<fold>" to get the base id shared by all folds.
            base_id = re.sub(r"-\d+$", "", model_id)
            plot_cv_individual_fold_curves(base_id, fold_num, log_path="logs", out_dir=out_dir.parent)
        return

    # Fallback: plain matplotlib
    csv = Path("logs") / model_id / "version_0" / "metrics.csv"
    if not csv.exists():
        print(f"  metrics.csv not found at {csv}, skipping loss curves.")
        return

    import pandas as pd
    df = pd.read_csv(csv).sort_values(["epoch", "step"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for col in ["train_loss", "val_loss", "val_mse_(weighted)"]:
        if col not in df:
            continue
        sub = df[["epoch", col]].dropna()
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(sub["epoch"], sub[col])
        ax.set_xlabel("Epoch")
        ax.set_ylabel(col)
        ax.set_title(col)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"{col}{PLOT_TYPE}", dpi=150)
        plt.close(fig)
        print(f"  Saved {col} plot.")


# ---------------------------------------------------------------------------
# Sample plots
# ---------------------------------------------------------------------------

def _unnorm(tensor, means, stds):
    """Unnormalise a (C, H, W) tensor using per-channel means/stds."""
    m = torch.as_tensor(means, dtype=tensor.dtype).view(-1, 1, 1)
    s = torch.as_tensor(stds,  dtype=tensor.dtype).view(-1, 1, 1)
    return tensor * s + m


def plot_sample(pred: torch.Tensor, target: torch.Tensor,
                sample_idx: int, out_dir: Path, diff_means, diff_stds) -> None:
    """
    Plot predicted and target state changes side by side for all variables at
    the 500 hPa level (index 1 out of [450, 500, 550]).

    pred/target: (15, lon, lat) normalised state-change tensors.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_un   = _unnorm(pred,   diff_means, diff_stds).cpu().numpy()
    target_un = _unnorm(target, diff_means, diff_stds).cpu().numpy()

    level_idx = 1  # 500 hPa
    ncols = len(VARS)
    fig, axes = plt.subplots(3, ncols, figsize=(4 * ncols, 9))
    fig.suptitle(f"Sample {sample_idx} — predicted vs target state change (500 hPa)", fontsize=12)

    for j, var in enumerate(VARS):
        ch = j * 3 + level_idx  # channel index: (var lev) ordering
        p = pred_un[ch]          # (lon, lat)
        t = target_un[ch]
        vmin = min(p.min(), t.min())
        vmax = max(p.max(), t.max())
        cmap = COLOR_BARS.get(var, "coolwarm") if COLOR_BARS else "coolwarm"

        for row, (data, title) in enumerate([(p, "Pred"), (t, "Target"), (p - t, "Diff")]):
            ax = axes[row, j]
            if row < 2:
                im = ax.imshow(data.T, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", aspect="auto")
            else:
                absmax = np.abs(p - t).max()
                im = ax.imshow(data.T, cmap="bwr", vmin=-absmax, vmax=absmax, origin="lower", aspect="auto")
            ax.set_title(f"{VAR_NAMES.get(var, var)} {title}")
            ax.axis("off")
            fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)

    fig.tight_layout()
    path = out_dir / f"sample_{sample_idx}{PLOT_TYPE}"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(model_id: str, n_samples: int, out_dir: Path,
         dataset_path: Path | None, min_year: str | None, max_year: str | None,
         fold_num: int | None = None) -> None:
    model_dir = Path("models") / model_id
    out_dir   = out_dir / model_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Loss curves --------------------------------------------------------
    print("Plotting loss curves ...")
    plot_loss_curves(model_id, out_dir, fold_num=fold_num)

    # --- Load model ---------------------------------------------------------
    print("Loading model ...")
    cfg = OmegaConf.load(model_dir / "config.yaml")

    # Allow overriding dataset location and year range from the command line
    # so the script can be run locally with a small local precomputed dataset
    # even when the training config points to an HPC path.
    if dataset_path is not None:
        cfg.dataset.path = str(dataset_path)
        print(f"  Overriding dataset path → {dataset_path}")
    if min_year is not None:
        cfg.dataset.min_year = int(min_year)
        print(f"  Overriding min_year → {min_year}")
    if max_year is not None:
        cfg.dataset.max_year = int(max_year)
        print(f"  Overriding max_year → {max_year}")

    cfg.model.dims = cfg.dataset.dims

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DiffusionModel(cfg)
    state_dict = torch.load(model_dir / "best-val_loss-weights.pt", map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print(f"  Loaded weights from {model_dir / 'best-val_loss-weights.pt'} (device={device})")

    # --- Load dataset -------------------------------------------------------
    print("Loading dataset ...")
    dataset = DatasetRegistry.create(cfg.dataset)
    print(f"  Dataset size: {len(dataset)} samples")
    diff_means = torch.as_tensor(dataset.diff_means, device=device)
    diff_stds  = torch.as_tensor(dataset.diff_stds, device=device)

    # --- Run inference on n_samples ----------------------------------------
    print(f"Running inference on {n_samples} sample(s) ...")
    indices = np.linspace(0, len(dataset) - 1, n_samples, dtype=int)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            cond, target = dataset[int(idx)]
            cond   = cond.unsqueeze(0).to(device)
            target = target.to(device)

            pred = model.sample_loop(batch_size=1, conditionals=cond)[0]  # (C_out, H, W)

            plot_sample(pred, target, sample_idx=int(idx),
                        out_dir=out_dir,
                        diff_means=diff_means,
                        diff_stds=diff_stds)

    print(f"\nAll outputs saved to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id",      type=str,  required=True,
                        help="Model run ID, e.g. tiny-geqcd")
    parser.add_argument("--n-samples",     type=int,  default=3,
                        help="Number of samples to visualise (default: 3)")
    parser.add_argument("--out-dir",       type=Path, default=Path("reports/inspect"),
                        help="Output directory for plots")
    parser.add_argument("--dataset-path",  type=Path, default=None,
                        help="Override dataset path from config (for local use)")
    parser.add_argument("--min-year",      type=str,  default=None,
                        help="Override min_year from config")
    parser.add_argument("--max-year",      type=str,  default=None,
                        help="Override max_year from config")
    parser.add_argument("--fold-num",      type=int,  default=None,
                        help="If set, also plot each fold's loss curve separately "
                             "(model-id's base, i.e. without the trailing '-<fold>', "
                             "is expected to have folds 1..fold-num under models/ and logs/)")
    args = parser.parse_args()
    main(args.model_id, args.n_samples, args.out_dir,
         args.dataset_path, args.min_year, args.max_year, args.fold_num)
