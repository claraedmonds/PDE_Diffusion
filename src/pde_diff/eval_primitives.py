"""
Shared low-level plotting primitives and style/constants used by both
pre-training dataset-exploration code (pde_diff.visualize) and post-training
inference/evaluation code (pde_diff.eval.*).

Kept as a single dependency-light module so neither side needs to import the
other.
"""
import os
import warnings
from pathlib import Path

from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as fm
import matplotlib.ticker as ticker

import numpy as np

# Anchor the font path to the repo root regardless of the caller's CWD
# (src/pde_diff/eval_primitives.py -> pde_diff -> src -> repo root).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_FONT_PATH = _REPO_ROOT / "times.ttf"
try:
    fm.fontManager.addfont(str(_FONT_PATH))
    plt.rcParams["font.family"] = "Times New Roman"
except Exception as e:
    warnings.warn(f"Could not load font from {_FONT_PATH}: {e}. Falling back to default font.")

try:
    plt.style.use("pde_diff.custom_style")
except Exception as e:
    warnings.warn(f"Could not load matplotlib style 'pde_diff.custom_style': {e}")
mpl.rcParams["mathtext.default"] = "regular"

# Used for plotting samples:
EXTENT_FULL = [0.0, 359.25, 90.0, -90.0]
EXTENT_SUBSET = [0.0, 359.25, 69.75, 46.5]
PLOT_TYPE = ".png"

VAR_NAMES = {
    "u": "u",
    "v": "v",
    "t": "T",
    "z": r"$\Phi$",
    "pv": r"$q_E$",
}

VAR_UNITS = {
    "u": r"$m \cdot s^{-1}$",
    "v": r"$m \cdot s^{-1}$",
    "t": r"$K$",
    "z": r"$m^2 \cdot s^{-2}$",
    "pv": r"$K\cdot m^2 \cdot kg^{-1} \cdot s^{-1}$",
}

colors = ["brown", "white", "teal"]
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
colors = ["teal", "white", "brown"]
custom_cmap2 = LinearSegmentedColormap.from_list("custom_cmap", colors)

COLOR_BARS = {
    "u": "RdBu",
    "v": "RdBu",
    "t": "coolwarm",
    "z": custom_cmap2,
    "pv": custom_cmap,
}


def visualize_era5_sample(data_sample, variable, level=500, big_data_sample=None, sample_idx=None, dir=Path("./reports/figures/samples"), color_bar_limit=None):
    """
    Visualize a sample from the ERA5 dataset.

    Args:
        data_sample (np.ndarray): The data sample to visualize.
        variable (str): Variable to visualize (e.g., 'temperature').
        level (int): Pressure level to focus on (default: 500hPa).
        big_data_sample (np.ndarray, optional): The entire dataset for context.
        sample_idx (int, optional): Index of the sample (for title purposes).
        color_bar_limit (tuple, optional): Tuple of (vmin, vmax) for color bar limits.
    """
    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    if big_data_sample is not None:
        vmin = min(big_data_sample.min(), data_sample.min())
        vmax = max(big_data_sample.max(), data_sample.max())

        ax.imshow(big_data_sample.T, cmap=COLOR_BARS.get(variable, 'coolwarm'), extent=EXTENT_FULL, origin='lower', vmin=vmin, vmax=vmax, alpha=0.5)
        ax.set_xlim(EXTENT_FULL[0], EXTENT_FULL[1])
        ax.set_ylim(EXTENT_FULL[3], EXTENT_FULL[2])

        ax.imshow(data_sample.T, cmap=COLOR_BARS.get(variable, 'coolwarm'), extent=EXTENT_SUBSET, origin='lower', vmin=vmin, vmax=vmax)
        rect = patches.Rectangle(
            (EXTENT_SUBSET[0], EXTENT_SUBSET[2]),
            EXTENT_SUBSET[1] - EXTENT_SUBSET[0],
            EXTENT_SUBSET[3] - EXTENT_SUBSET[2],
            linewidth=2,
            edgecolor='black',
            facecolor='none'
        )
        ax.add_patch(rect)
        fig.colorbar(ax.images[-1], ax=ax, label=f"{VAR_UNITS.get(variable, variable)}")
    else:
        ax.imshow(data_sample.T, cmap=COLOR_BARS.get(variable, 'coolwarm'), extent=EXTENT_SUBSET, origin='lower', vmin=color_bar_limit[0] if color_bar_limit else None, vmax=color_bar_limit[1] if color_bar_limit else None)
        fig.colorbar(ax.images[-1], ax=ax, shrink=0.4, location="bottom")
        ax.set_xlim(EXTENT_SUBSET[0], EXTENT_SUBSET[1])
        ax.set_ylim(EXTENT_SUBSET[3], EXTENT_SUBSET[2])

    ax.set_title(f"{VAR_NAMES.get(variable, variable)} at {level} hPa")
    plt.tight_layout()
    plot_path = f"era5_sample{sample_idx if sample_idx is not None else ''}_{variable}_{level}hPa"
    plot_path += "_full" if big_data_sample is not None else ""
    plot_path += PLOT_TYPE
    plot_path = os.path.join(dir, plot_path)
    plt.savefig(plot_path, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f"Saved visualization to {plot_path}")


def visualize_era5_sample_cropped(
    big_data_sample,
    variable,
    level=500,
    sample_idx=None,
    dir=Path("./reports/figures/samples"),
    limits=None,
    title=None,
):
    """
    Visualize and save a cropped ERA5 sample image.

    Crop:
    - 20 pixels in from the left
    - full height
    - width = 2 * height
    """
    fig, ax = plt.subplots(figsize=(4, 2), dpi=200)

    vmin = limits[0] if limits else None
    vmax = limits[1] if limits else None

    cropped_sample = big_data_sample.T[:, 20:84]

    extent = [20.0, 84, 69.75, 46.5]

    ax.imshow(
        cropped_sample,
        cmap=COLOR_BARS.get(variable, "coolwarm"),
        origin="lower",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_yticks([])
    ax.set_ylabel("5 h", rotation=0, va="center", labelpad=10)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[3], extent[2])
    ax.set_title(title)

    plot_path = (
        f"era5_sample{sample_idx if sample_idx is not None else ''}_"
        f"{variable}_{level}hPa_cropped_only{PLOT_TYPE}"
    )
    plot_path = dir / plot_path

    plt.savefig(plot_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"Saved cropped visualization to {plot_path}")


def save_colorbar(
    variable,
    vmin,
    vmax,
    suffix="",
    dir=Path("./reports/figures/samples"),
):
    fig, ax = plt.subplots(figsize=(0.3, 2))

    cmap = COLOR_BARS.get(variable, "coolwarm")
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax,
        label=f"{VAR_UNITS.get(variable, variable)}"
    )

    if variable == "z" or variable == "pv":
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))

        cbar.formatter = formatter
        cbar.update_ticks()

    plot_path = dir / f"colorbar_{variable}{suffix}{PLOT_TYPE}"
    plt.savefig(plot_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    print(f"Saved colorbar to {plot_path}")


def visualize_era5_sample_full(big_data_sample, variable, level=500, sample_idx=None, dir=Path("./reports/figures/samples"), limits=None):
    """Visualize a sample from the ERA5 dataset at the full (uncropped) subset extent."""
    fig, ax = plt.subplots(figsize=(8, 4))
    vmin = limits[0] if limits else None
    vmax = limits[1] if limits else None
    ax.imshow(big_data_sample.T, cmap=COLOR_BARS.get(variable, 'coolwarm'), extent=EXTENT_SUBSET, origin='lower', vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(EXTENT_SUBSET[0], EXTENT_SUBSET[1])
    ax.set_ylim(EXTENT_SUBSET[3], EXTENT_SUBSET[2])

    plot_path = f"era5_sample{sample_idx if sample_idx is not None else ''}_{variable}_{level}hPa_full_only"
    plot_path += PLOT_TYPE
    plot_path = dir / plot_path
    plt.savefig(plot_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"Saved visualization to {plot_path}")


def find_metrics_csv(model_id: str, log_path: str = "logs") -> Path:
    """Locate the most recent `version_N/metrics.csv` for a given model id.

    Lightning's CSVLogger increments the version directory (version_0,
    version_1, ...) every time a run is (re)started under the same log
    name, so `version_0` is not always where the latest metrics live —
    e.g. after resubmitting a training job. Raises FileNotFoundError with a
    clear message if no version directory exists at all.
    """
    base = Path(log_path) / model_id
    version_dirs = sorted(
        (d for d in base.glob("version_*") if d.is_dir()),
        key=lambda d: int(d.name.split("_")[-1]),
    )
    if not version_dirs:
        raise FileNotFoundError(f"No version_*/metrics.csv found under {base}")
    return version_dirs[-1] / "metrics.csv"


def plot_training_metrics(model_id, out_dir=Path("./reports/figures")):
    import pandas as pd
    df = pd.read_csv(find_metrics_csv(model_id)).sort_values(["epoch", "step"])
    save_dir = Path(out_dir) / model_id
    save_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("train_loss", True, "Train Loss vs Epoch"),
        ("val_loss", True, "Validation Loss vs Epoch"),
        ("val_mse", False, "Validation MSE vs Epoch"),
    ]

    for col, logy, title in metrics:
        if col not in df:
            continue
        sub = df[["epoch", col]].dropna()
        if sub.empty:
            continue

        ax = sub.plot(x="epoch", y=col, legend=False, figsize=(8, 4.5))
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(col)
        if logy:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.figure.tight_layout()
        ax.figure.savefig(save_dir / f"{col}{PLOT_TYPE}", dpi=150)
        plt.close(ax.figure)

    if {"val_loss", "val_mse"}.issubset(df.columns):
        sub = df[["epoch", "val_loss", "val_mse"]].dropna()
        if not sub.empty:
            ax = sub.plot(x="epoch", y=["val_loss", "val_mse"], figsize=(8, 4.5))
            ax.set_title("Validation Metrics vs Epoch")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Value")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.figure.tight_layout()
            ax.figure.savefig(save_dir / f"val_metrics_combined{PLOT_TYPE}", dpi=150)
            plt.close(ax.figure)


def plot_cv_individual_fold_curves(
    model_id,
    fold_num,
    log_path="logs",
    out_dir=Path("./reports/figures"),
    metrics=("train_loss", "val_loss"),
    smooth_window=1,
):
    """
    Plot each fold's loss curve as a separate line on one shared-axes plot
    (one figure per metric), for a cross-validated model
    `{model_id}-1` .. `{model_id}-{fold_num}`.
    """
    import pandas as pd

    out_dir = Path(out_dir)
    save_dir = out_dir / model_id
    save_dir.mkdir(parents=True, exist_ok=True)

    def smooth(arr, window):
        if window <= 1:
            return arr
        kernel = np.ones(window) / window
        return np.convolve(arr, kernel, mode="valid")

    cmap = plt.get_cmap("tab10")

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        any_plotted = False

        for fold in range(1, fold_num + 1):
            try:
                csv_path = find_metrics_csv(f"{model_id}-{fold}", log_path=log_path)
            except FileNotFoundError as e:
                print(f"  Skipping fold {fold}: {e}")
                continue

            df = (
                pd.read_csv(csv_path)
                .apply(pd.to_numeric, errors="coerce")
                .dropna(subset=["step"])
                .sort_values("step")
            )
            if metric not in df.columns:
                continue

            sub = df[["epoch", metric]].dropna()
            if sub.empty:
                continue

            epochs = sub["epoch"].values
            vals = sub[metric].values
            if smooth_window > 1:
                vals = smooth(vals, smooth_window)
                epochs = epochs[: len(vals)]

            ax.plot(epochs, vals, label=f"Fold {fold}", color=cmap(fold % 10))
            any_plotted = True

        if not any_plotted:
            plt.close(fig)
            print(f"  No data found for metric '{metric}', skipping plot.")
            continue

        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} per fold — {model_id}")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=True, fancybox=True, framealpha=0.9)
        fig.tight_layout()

        save_path = save_dir / f"{metric}_per_fold{PLOT_TYPE}"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {save_path}")
