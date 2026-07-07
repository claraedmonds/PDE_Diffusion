import os
from pathlib import Path
import yaml

from matplotlib.colors import LinearSegmentedColormap, LogNorm
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as fm
import matplotlib.ticker as ticker

import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import torch

from pde_diff.data.datasets import ERA5Dataset
from pde_diff.model import DiffusionModel
from pde_diff.loss import DarcyLoss
from pde_diff.data.utils import split_dataset

# Path to your TTF file
font_path = "./times.ttf"
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'Times New Roman'

plt.style.use("pde_diff.custom_style")
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

colors = ["brown", "white", "teal"]  # Transition from blue to white to red
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
colors = ["teal", "white", "brown"]  # Transition from blue to white to red
custom_cmap2 = LinearSegmentedColormap.from_list("custom_cmap", colors)

COLOR_BARS = {
    "u": "RdBu",
    "v": "RdBu",
    "t": "coolwarm",
    "z": custom_cmap2,
    "pv": custom_cmap,
}

# Custom colour palette (different from default Matplotlib cycle)
diffusion_colors = ("#8800FF","#5900A7")
pidm_colors = {
    'c1e1': "#E32D00",
    'c1e2': "#1A4DAC",
    'c1e3': "#2B9A0C",
    'c1e2_pv': "#BF0251",
    'c1e2_gw': "#D98123",
}
train_loss_pidm_colors = {
    "c1e1": "#F18A70",
    "c1e2": "#4E78A0",
    "c1e3": "#88DB72",
    "c1e2_pv": "#DA729D",
    "c1e2_gw": "#D97223",
}

val_loss_pidm_colors = {
    "c1e1": "#E32D00",
    "c1e2": "#1A4DAC",
    "c1e3": "#2B9A0C",
    "c1e2_pv": "#BF0251",
    "c1e2_gw": "#D98123",
}
train_loss_diff_colors = ("#B977F2","#6E5386")
val_loss_diff_colors   = ("#8800FF","#5900A7")

model_id_to_name={
        "c1e1": r"c=1e-1",
        "c1e2": r"c=1e-2",
        "c1e3": r"c=1e-3",
        "c1e2_pv": r"$\mathcal{R}_1$: c=1e-2",
        "c1e2_gw": r"$\mathcal{R}_2$: c=1e-2",
    }    

def plot_darcy_samples(model_1, model_2, model_id, out_dir=Path("./reports/figures")):
    save_dir = Path(out_dir) / model_id
    save_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.create({"c_residual": None})
    loss = DarcyLoss(cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def generate(model):
        samples = model.sample_loop(batch_size=1)
        loss_samples = loss.compute_residual_field_for_plot(samples.to(device))

        samples = samples.detach().cpu().numpy()
        loss_samples = loss_samples.detach().cpu().numpy()
        return samples, loss_samples

    # --- 1) Generate both ---
    samples1, res1 = generate(model_1)
    samples2, res2 = generate(model_2)

    # --- 2) Build a shared LogNorm for the residual plot (axs[2]) ---
    # LogNorm requires strictly positive values; clamp zeros/negatives.
    eps = 1e-12
    r1 = np.clip(res1[0], eps, None)
    r2 = np.clip(res2[0], eps, None)

    shared_vmin = float(min(r1.min(), r2.min()))
    shared_vmax = float(max(r1.max(), r2.max()))
    shared_norm = LogNorm(vmin=shared_vmin, vmax=shared_vmax)

    def plot_one(samples, res, tag):
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))

        # Plot #1
        im0 = axs[0].imshow(samples[0, 0], cmap="magma")
        axs[0].set_title(r"Permeability - $K$")
        axs[0].axis("off")
        fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

        # Plot #2
        im1 = axs[1].imshow(np.rot90(samples[0, 1], k=3), cmap="magma")
        axs[1].set_title(r"Pressure - $P$")
        axs[1].axis("off")
        fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

        # Plot #3 (shared scale!)
        res_plot = np.clip(res[0], eps, None)
        breakpoint()
        im2 = axs[2].imshow(res_plot, cmap="magma", norm=shared_norm)
        axs[2].set_title(r"Residual - $\mathcal{R}_{\text{MAE}}$")
        axs[2].axis("off")
        fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

        out_path = save_dir / f"samples_{tag}{PLOT_TYPE}"
        plt.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved samples to {out_path}")

    # --- 3) Plot two figures, same residual norm ---
    plot_one(samples1, res1, tag="model1")
    plot_one(samples2, res2, tag="model2")


def plot_training_metrics(model_id, out_dir=Path("./reports/figures")):
    df = pd.read_csv(Path("./logs") / model_id / "version_0" / "metrics.csv").sort_values(["epoch", "step"])
    save_dir = Path(out_dir) / model_id
    save_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("train_loss", True,  "Train Loss vs Epoch"),
        ("val_loss", True,  "Validation Loss vs Epoch"),
        ("val_mse", False, "Validation MSE vs Epoch"),
        ("val_darcy_residual", True, "Validation Darcy Residual vs Epoch"),
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
    Plot each fold's loss curve separately (one line per fold, no averaging
    across folds), for a cross-validated model `{model_id}-1` .. `{model_id}-{fold_num}`.
    """
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
            csv_path = Path(log_path) / f"{model_id}-{fold}" / "version_0" / "metrics.csv"
            if not csv_path.exists():
                print(f"  Skipping fold {fold}: {csv_path} not found.")
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


def get_data_sample(dataset, sample_idx, variable, level=500):
    """
    Extract data for visualization from the ERA5 dataset.

    """
    # Extract the sample
    sample = dataset[sample_idx]
    inputs, _ = sample

    # Find the index of the variable and level
    var_idx = dataset.atmospheric_features.index(variable)
    level_idx = np.where(dataset.pressure_levels == level)[0][0]

    # Extract the data for the variable and level
    data = inputs[var_idx * len(dataset.pressure_levels) + level_idx]
    return data

def get_time_series(dataset, variable, level=500, coords=(12.568, 55.676)):
    """
    Extract a time series of a variable at a specific pressure level from the ERA5 dataset.

    Args:
        dataset (ERA5Dataset): The ERA5 dataset.
        variable (str): Variable to extract (e.g., 'temperature').
        level (int): Pressure level to focus on (default: 500hPa).
        coords (tuple): Tuple of (longitude, latitude) to extract the time series from. Default is (12.568, 55.676) (Copenhagen).
    """
    level_idx = np.where(dataset.pressure_levels == level)[0][0]
    lon, lat = coords
    lon_idx = np.argmin(np.abs(dataset.grid_lon - lon))
    lat_idx = np.argmin(np.abs(dataset.grid_lat - lat))
    print(dataset.data[variable].shape)
    time_series = dataset.data[variable][:, level_idx, lon_idx, lat_idx]
    return time_series


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

    # Plot the data
    fig, ax = plt.subplots(figsize=(6,3))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    if big_data_sample is not None:
        vmin = min(big_data_sample.min(), data_sample.min())
        vmax = max(big_data_sample.max(), data_sample.max())

        # Draw entire dataset first
        ax.imshow(big_data_sample.T, cmap=COLOR_BARS.get(variable, 'coolwarm'), extent=EXTENT_FULL, origin='lower',vmin=vmin, vmax=vmax, alpha=0.5)
        ax.set_xlim(EXTENT_FULL[0], EXTENT_FULL[1])
        ax.set_ylim(EXTENT_FULL[3], EXTENT_FULL[2])

        # Draw subset on top in correct place
        ax.imshow(data_sample.T, cmap=COLOR_BARS.get(variable, 'coolwarm'), extent=EXTENT_SUBSET, origin='lower',vmin=vmin, vmax=vmax)
        rect = patches.Rectangle(
            (EXTENT_SUBSET[0], EXTENT_SUBSET[2]),           # bottom-left corner (lon_min, lat_min)
            EXTENT_SUBSET[1]-EXTENT_SUBSET[0],              # width (lon_max - lon_min)
            EXTENT_SUBSET[3]-EXTENT_SUBSET[2],              # height (lat_max - lat_min)
            linewidth=2,
            edgecolor='black',                         # border color
            facecolor='none'                           # transparent fill
        )
        ax.add_patch(rect)
        fig.colorbar(ax.images[-1], ax=ax, label=f"{VAR_UNITS.get(variable, variable)}")
    else:
        ax.imshow(data_sample.T, cmap=COLOR_BARS.get(variable, 'coolwarm'), extent=EXTENT_SUBSET, origin='lower', vmin=color_bar_limit[0] if color_bar_limit else None, vmax=color_bar_limit[1] if color_bar_limit else None)
        fig.colorbar(ax.images[-1], ax=ax, shrink=0.4, location="bottom") #, label=f"{VAR_UNITS.get(variable, variable)}")
        ax.set_xlim(EXTENT_SUBSET[0], EXTENT_SUBSET[1])
        ax.set_ylim(EXTENT_SUBSET[3], EXTENT_SUBSET[2])

    ax.set_title(f"{VAR_NAMES.get(variable, variable)} at {level} hPa")
    plt.tight_layout()
    plot_path = f"era5_sample{sample_idx if sample_idx is not None else ''}_{variable}_{level}hPa"
    plot_path += "_full" if big_data_sample is not None else ""
    plot_path += PLOT_TYPE
    plot_path = os.path.join(dir, plot_path)
    plt.savefig(plot_path, bbox_inches='tight', pad_inches=0.05)
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
    print(f"{cropped_sample.min()}, {cropped_sample.max()}")

    ax.imshow(
        cropped_sample,
        cmap=COLOR_BARS.get(variable, "coolwarm"),
        origin="lower",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
    )

    #ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel("5 h", rotation=0, va="center", labelpad=10)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[3], extent[2])
    ax.set_title(title)
    
    # Save cropped image
    plot_path = (
        f"era5_sample{sample_idx if sample_idx is not None else ''}_"
        f"{variable}_{level}hPa_cropped_only{PLOT_TYPE}"
    )
    plot_path = dir / plot_path

    plt.savefig(plot_path, bbox_inches='tight', pad_inches=0)
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

    cbar= fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax,
        label=f"{VAR_UNITS.get(variable, variable)}"
    )

    if variable == "z" or variable == "pv":
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))  # always use scientific notation

        cbar.formatter = formatter
        cbar.update_ticks()

    plot_path = dir / f"colorbar_{variable}{suffix}{PLOT_TYPE}"
    plt.savefig(plot_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    print(f"Saved colorbar to {plot_path}")

def visualize_era5_sample_full(big_data_sample, variable, level=500, sample_idx=None, dir=Path("./reports/figures/samples"), limits=None):
    """
    Visualize a sample from the ERA5 dataset.

    Args:
        data_sample (np.ndarray): The data sample to visualize.
        variable (str): Variable to visualize (e.g., 'temperature').
        level (int): Pressure level to focus on (default: 500hPa).
        big_data_sample (np.ndarray, optional): The entire dataset for context.
        sample_idx (int, optional): Index of the sample (for title purposes).
    """
    # Plot the data
    fig, ax = plt.subplots(figsize=(8,4))
    vmin = limits[0] if limits else None
    vmax = limits[1] if limits else None
    # Draw entire data
    ax.imshow(big_data_sample.T, cmap=COLOR_BARS.get(variable, 'coolwarm'), extent=EXTENT_SUBSET, origin='lower', vmin=vmin, vmax=vmax)
    #remove all ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(EXTENT_SUBSET[0], EXTENT_SUBSET[1])
    ax.set_ylim(EXTENT_SUBSET[3], EXTENT_SUBSET[2])

    plot_path = f"era5_sample{sample_idx if sample_idx is not None else ''}_{variable}_{level}hPa_full_only"
    plot_path += PLOT_TYPE
    plot_path = dir / plot_path
    plt.savefig(plot_path, bbox_inches='tight', pad_inches=0)
    print(f"Saved visualization to {plot_path}")

def visualize_noise_schedule(scheduler, data_sample, variable, level=500, sample_idx=None, dir=Path("./reports/figures/samples"), steps= 10):
    """
    Visualize a sample from the ERA5 dataset.

    Args:
        data_sample (np.ndarray): The data sample to visualize.
        variable (str): Variable to visualize (e.g., 'temperature').
        level (int): Pressure level to focus on (default: 500hPa).
        sample_idx (int, optional): Index of the sample (for title purposes).
    """
    # Plot the data
    fig, ax = plt.subplots(figsize=(data_sample.shape[1] / 5, data_sample.shape[0] / 5))
    ax.imshow(data_sample.T, cmap=COLOR_BARS.get(variable, 'coolwarm'), extent=EXTENT_SUBSET, origin='lower')
    ax.set_xlim(EXTENT_SUBSET[0], EXTENT_SUBSET[1])
    ax.set_ylim(EXTENT_SUBSET[3], EXTENT_SUBSET[2])
    ax.axis('off')  # Remove axes
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0) #Remove margins

    # Save the plot
    plot_path = f"era5_sample{sample_idx if sample_idx is not None else ''}_{variable}_{level}hPa_noise_{0}"
    plot_path += PLOT_TYPE
    plot_path = dir / plot_path
    plt.savefig(plot_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"Saved visualization to {plot_path}")

    for t in range(100,scheduler.config.num_train_timesteps+1, scheduler.config.num_train_timesteps//steps):
        noised_sample = scheduler.add_noise(torch.tensor(data_sample).unsqueeze(0).unsqueeze(0).float(), torch.randn_like(torch.tensor(data_sample).unsqueeze(0).unsqueeze(0).float()), torch.tensor([t-1]))
        noised_sample = noised_sample.squeeze().squeeze().numpy()
        fig, ax = plt.subplots(figsize=(noised_sample.shape[1] / 5, noised_sample.shape[0] / 5))
        ax.imshow(noised_sample.T, cmap=COLOR_BARS.get(variable, 'coolwarm'), extent=EXTENT_SUBSET, origin='lower')
        ax.set_xlim(EXTENT_SUBSET[0], EXTENT_SUBSET[1])
        ax.set_ylim(EXTENT_SUBSET[3], EXTENT_SUBSET[2])
        ax.axis('off')  # Remove axes
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0) #Remove margins

        # Save the plot
        plot_path = f"era5_sample{sample_idx if sample_idx is not None else ''}_{variable}_{level}hPa_noise_{t}"
        plot_path += PLOT_TYPE
        plot_path = dir / plot_path
        plt.savefig(plot_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print(f"Saved visualization to {plot_path}")


def visualize_time_series(dataset, variable, level=500, dir=Path("./reports/figures/time_series"), coords=(12.568, 55.676)):
    """
    Visualize a time series of a variable at a specific pressure level from the ERA5 dataset.

    Args:
        dataset (ERA5Dataset): The ERA5 dataset.
        variable (str): Variable to visualize (e.g., 'temperature').
        level (int): Pressure level to focus on (default: 500hPa).
        dir (Path): Directory to save the plots.
        coords (tuple): Tuple of (longitude, latitude) to extract the time series from. Default is (12.568, 55.676) (Copenhagen).
    """

    dir.mkdir(parents=True, exist_ok=True)
    lon, lat = coords

    time_series = np.array(get_time_series(dataset, variable, level, coords))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_series, label=f"{VAR_NAMES.get(variable, variable)} at {level} hPa")
    ax.set_title(f"Time Series of {VAR_NAMES.get(variable, variable)} at {level} hPa\nLocation: Lon {lon}, Lat {lat}")
    ax.set_xlabel("Time Step")
    ax.set_ylabel(VAR_NAMES.get(variable, variable))
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plot_path = dir / f"time_series_{variable}_{level}hPa_lon{lon}_lat{lat}{PLOT_TYPE}"
    plt.savefig(plot_path, bbox_inches='tight', pad_inches=0.05)
    print(f"Saved time series plot to {plot_path}")

def _to_2d(x):
    """Accepts (T,), (N,T), list of (T,), list of list; returns (N,T) float array."""
    if x is None:
        return None
    if isinstance(x, (list, tuple)) and len(x) > 0 and isinstance(x[0], (list, tuple, np.ndarray)):
        x = np.array(x, dtype=float)
    else:
        x = np.array(x, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array-like, got shape {x.shape}")
    return x

def plot_darcy_val_metrics(model_id_1, model_id_2, fold_num, log_path, out_dir, smooth_window=10):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path

    # --- Global styling tweaks (do NOT set figure.figsize here; we set it per-figure) ---
    plt.rcParams.update({
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    })

    # Custom colour palette (different from default Matplotlib cycle)
    diffusion_color = "#971A7C"
    diffusion_ci    = "#ED63CF"
    pidm_color      = "#BA090C"
    pidm_ci         = "#F77C7E"
    train_loss_pidm_color = "#0239C5"
    train_loss_pidm_ci    = "#3B60BE"
    val_loss_pidm_color   = "#028EC5"
    val_loss_pidm_ci      = "#4497B8"
    train_loss_diff_color = "#E2901C"
    train_loss_diff_ci    = "#D19541"
    val_loss_diff_color   = "#D17321"
    val_loss_diff_ci      = "#CE9563"

    def moving_average_2d(arr, window):
        """Apply moving average along the time axis for a 2D array [fold, time]."""
        if window <= 1:
            return arr
        kernel = np.ones(window) / window
        return np.apply_along_axis(
            lambda m: np.convolve(m, kernel, mode="valid"), axis=1, arr=arr
        )

    def _load_model_stats(model_id, smooth_window=1):
        residual_errors = []
        weighted_mse_errors = []
        train_loss = []
        val_loss = []
        steps = None

        for fold in range(1, fold_num + 1):
            current_model_id = f"{model_id}-{fold}"
            csv_path = Path(log_path) / current_model_id / "version_0" / "metrics.csv"

            df = (
                pd.read_csv(csv_path)
                .apply(pd.to_numeric, errors="ignore")
                .dropna(subset=["step"])
                .sort_values("step")
            )

            if steps is None:
                steps = df["step"].values

            residual_errors.append(df["val_darcy_residual"].dropna().values)
            weighted_mse_errors.append(df["val_mse_(weighted)"].dropna().values)
            train_loss.append(df["train_loss"].dropna().values)
            val_loss.append(df["val_loss"].dropna().values)

        residual_errors = np.array(residual_errors)      # shape: [fold, time]
        weighted_mse_errors = np.array(weighted_mse_errors)
        train_loss = np.array(train_loss)
        val_loss = np.array(val_loss)

        # --- Smooth over time (epochs) ---
        residual_errors = moving_average_2d(residual_errors, smooth_window)
        weighted_mse_errors = moving_average_2d(weighted_mse_errors, smooth_window)
        train_loss = moving_average_2d(train_loss, smooth_window)
        val_loss = moving_average_2d(val_loss, smooth_window)

        n = residual_errors.shape[0]

        res_mean = residual_errors.mean(axis=0)
        res_std = residual_errors.std(axis=0)
        res_low = res_mean - 1.96 * res_std / np.sqrt(n)
        res_high = res_mean + 1.96 * res_std / np.sqrt(n)

        mse_mean = weighted_mse_errors.mean(axis=0)
        mse_std = weighted_mse_errors.std(axis=0)
        mse_low = mse_mean - 1.96 * mse_std / np.sqrt(n)
        mse_high = mse_mean + 1.96 * mse_std / np.sqrt(n)

        train_loss_mean = train_loss.mean(axis=0)
        train_loss_std = train_loss.std(axis=0)
        val_loss_mean = val_loss.mean(axis=0)
        val_loss_std = val_loss.std(axis=0)

        train_loss_low = train_loss_mean - 1.96 * train_loss_std / np.sqrt(n)
        train_loss_high = train_loss_mean + 1.96 * train_loss_std / np.sqrt(n)
        val_loss_low = val_loss_mean - 1.96 * val_loss_std / np.sqrt(n)
        val_loss_high = val_loss_mean + 1.96 * val_loss_std / np.sqrt(n)

        epochs = res_mean.shape[0]

        return {
            "epochs": epochs,
            "res_mean": res_mean,
            "res_low": res_low,
            "res_high": res_high,
            "mse_mean": mse_mean,
            "mse_low": mse_low,
            "mse_high": mse_high,
            "train_loss_mean": train_loss_mean,
            "train_loss_low": train_loss_low,
            "train_loss_high": train_loss_high,
            "val_loss_mean": val_loss_mean,
            "val_loss_low": val_loss_low,
            "val_loss_high": val_loss_high,
        }

    # ---------------- Output folder ----------------
    out_dir = Path(out_dir)
    plot_folder = out_dir / f"{model_id_1}_vs_{model_id_2}_darcy_val_metrics"
    plot_folder.mkdir(parents=True, exist_ok=True)

    stats1 = _load_model_stats(model_id_1, smooth_window)
    stats2 = _load_model_stats(model_id_2, smooth_window)

    epochs = min(stats1["epochs"], stats2["epochs"])
    x = np.arange(epochs)

    def save_fig(fig, filename):
        path = plot_folder / filename
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")

    # ---------------- Plot 1: Residual ----------------
    fig, ax = plt.subplots(figsize=(4.1, 4.1))

    ax.plot(x, stats1["res_mean"][:epochs], label="Diffusion mean", linewidth=2.2, color=diffusion_color)
    ax.fill_between(x, stats1["res_low"][:epochs], stats1["res_high"][:epochs],
                    alpha=0.3, color=diffusion_ci, label="Diffusion 95% CI")

    ax.plot(x, stats2["res_mean"][:epochs], label="PIDM mean", linewidth=2.2, color=pidm_color)
    ax.fill_between(x, stats2["res_low"][:epochs], stats2["res_high"][:epochs],
                    alpha=0.3, color=pidm_ci, label="PIDM 95% CI")

    ax.set_xlabel(f"Epochs")
    ax.set_ylabel(r"$\mathcal{R}_{\text{MAE}}(\mathbf{x_0}) \sim p_\theta (\mathbf{x_0})$")
    ax.set_title("Darcy Sample Residual MAE")
    ax.set_yscale("log")
    ax.legend(frameon=True, fancybox=True, framealpha=0.9, loc="upper right")
    fig.tight_layout()
    save_fig(fig, "01_val_darcy_residual.png")

    # ---------------- Plot 2: Weighted MSE ----------------
    fig, ax = plt.subplots(figsize=(4.1, 4.1))

    ax.plot(x, stats1["mse_mean"][:epochs], label="Diffusion mean", linewidth=2.2, color=diffusion_color)
    ax.fill_between(x, stats1["mse_low"][:epochs], stats1["mse_high"][:epochs],
                    alpha=0.3, color=diffusion_ci, label="Diffusion 95% CI")

    ax.plot(x, stats2["mse_mean"][:epochs], label="PIDM mean", linewidth=2.2, color=pidm_color)
    ax.fill_between(x, stats2["mse_low"][:epochs], stats2["mse_high"][:epochs],
                    alpha=0.3, color=pidm_ci, label="PIDM 95% CI")

    ax.set_xlabel(f"Epochs")
    ax.set_ylabel(r"$\mathbb{E}_{t, \mathbf{x_0}}[\lambda_t \|\mathbf{x_0} - \hat{\mathbf{x}}_0 (\mathbf{x}_t,t) \|^2]$")
    ax.set_title("Validation Weighted MSE")
    ax.set_yscale("log")
    ax.legend(frameon=True, fancybox=True, framealpha=0.9, loc="upper right")
    fig.tight_layout()
    save_fig(fig, "02_val_weighted_mse.png")

    # ---------------- Plot 3: Train vs Val loss ----------------
    fig, ax = plt.subplots(figsize=(4.1, 4.1))

    # Diffusion
    ax.plot(x, stats1["train_loss_mean"][:epochs], label="Train loss mean (Diffusion)", linewidth=2.2, color=train_loss_diff_color)
    ax.fill_between(x, stats1["train_loss_low"][:epochs], stats1["train_loss_high"][:epochs],
                    alpha=0.3, color=train_loss_diff_ci, label="Train loss 95% CI (Diffusion)")

    ax.plot(x, stats1["val_loss_mean"][:epochs], label="Val loss mean (Diffusion)", linewidth=2.2, color=val_loss_diff_color)
    ax.fill_between(x, stats1["val_loss_low"][:epochs], stats1["val_loss_high"][:epochs],
                    alpha=0.3, color=val_loss_diff_ci, label="Val loss 95% CI (Diffusion)")

    # PIDM
    ax.plot(x, stats2["train_loss_mean"][:epochs], label="Train loss mean (PIDM)", linewidth=2.2, color=train_loss_pidm_color)
    ax.fill_between(x, stats2["train_loss_low"][:epochs], stats2["train_loss_high"][:epochs],
                    alpha=0.3, color=train_loss_pidm_ci, label="Train loss 95% CI (PIDM)")

    ax.plot(x, stats2["val_loss_mean"][:epochs], label="Val loss mean (PIDM)", linewidth=2.2, color=val_loss_pidm_color)
    ax.fill_between(x, stats2["val_loss_low"][:epochs], stats2["val_loss_high"][:epochs],
                    alpha=0.3, color=val_loss_pidm_ci, label="Val loss 95% CI (PIDM)")

    ax.set_xlabel(f"Epochs")
    ax.set_ylabel(
        r"$\mathbb{E}_{t, \mathbf{x_0}}[\lambda_t \|\mathbf{x_0} - \hat{\mathbf{x}}_0 (\mathbf{x}_t,t) \|^2] + "
        r"\frac{1}{2 \tilde{\Sigma}} || \mathcal{R}(\mathbf{x}_0^*(\mathbf{x}_t,t))||^2$"
    )
    ax.set_title("Train vs Validation loss")
    ax.set_yscale("log")
    ax.legend(frameon=True, fancybox=True, framealpha=0.9, loc="upper right")
    fig.tight_layout()
    save_fig(fig, "03_train_vs_val_loss.png")


def moving_average_2d(arr, window):
    """Apply moving average along the time axis for a 2D array [fold, time]."""
    if window <= 1:
        return arr
    kernel = np.ones(window) / window
    return np.apply_along_axis(
        lambda m: np.convolve(m, kernel, mode="valid"), axis=1, arr=arr
    )


def load_model_stats(
    model_id,
    smooth_window=1,
    fold_num=0,
    log_path="logs",
    data_type="era5",
):
    residual_errors = []
    weighted_mse_errors = []
    train_loss = []
    val_loss = []
    res1_pv_sample = []
    res2_geowind_sample = []
    steps = None

    # 5 independent lists (one per var)
    val_mse_var = [[] for _ in range(5)]

    era5_cols = [
        "val_era5_geo_wind_residual(norm)",
        "val_era5_planetary_residual(norm)",
    ]

    sample_cols = {
        "val_era5_sampled_planetary_residual(norm)": res1_pv_sample,
        "val_era5_sampled_geo_wind_residual(norm)": res2_geowind_sample,
    }

    def read_metrics(mid: str) -> pd.DataFrame:
        prev_run_exists = False
        if "retrain" in mid:
            # if retrain search for prev version, should be remove fold nr and "retrain"
            prev_mid = "-".join(mid.split("-")[:-2])+"-"+mid.split("-")[-1]
            prev_df = read_metrics(prev_mid)
            prev_run_exists = True
        csv_path = Path(log_path) / mid / "version_0" / "metrics.csv"
        df = pd.read_csv(csv_path).apply(pd.to_numeric, errors="coerce").dropna(subset=["step"]).sort_values("step")

        if prev_run_exists:
            last_epoch = prev_df["epoch"].max()
            df["epoch"]+=last_epoch
            return pd.concat([prev_df, df])
        return df

    def append_if(df: pd.DataFrame, col: str, target: list):
        if col in df.columns:
            vals = df[col].dropna().values
            if len(vals) > 0:
                target.append(vals)

    model_ids = (
        [model_id]
        if fold_num is None
        else [f"{model_id}-{f}" for f in range(1, fold_num + 1)]
    )

    # ------------------- Read + collect per-fold arrays -------------------
    for mid in model_ids:
        df = read_metrics(mid)

        steps = df["step"].values if steps is None else steps

        # residual_errors: ERA5 sum if ALL exist; else Darcy if exists; else nothing
        if all(c in df.columns for c in era5_cols):
            tmp = df[era5_cols].dropna(how="any").sum(axis=1).values
            if len(tmp) > 0:
                residual_errors.append(tmp)
        else:
            append_if(df, "val_darcy_residual", residual_errors)

        # sampled residuals (append only if column exists)
        for col, target in sample_cols.items():
            append_if(df, col, target)

        # always-collected series (append only if present; avoids KeyError)
        append_if(df, "val_mse_(weighted)", weighted_mse_errors)
        append_if(df, "train_loss", train_loss)
        append_if(df, "val_loss", val_loss)

        # val_mse_var_0..4 (weighted)
        for i in range(5):
            col = f"val_mse_var_{i}_(weighted)"
            append_if(df, col, val_mse_var[i])

    # ------------------- Trim everything to the shortest length -------------------
    series_lists = {
        "residual_errors": residual_errors,
        "weighted_mse_errors": weighted_mse_errors,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "res1_pv_sample": res1_pv_sample,
        "res2_geowind_sample": res2_geowind_sample,
    }
    for i in range(5):
        series_lists[f"val_mse_var_{i}"] = val_mse_var[i]

    nonempty = [lst for lst in series_lists.values() if len(lst) > 0]
    min_length = min((len(arr) for lst in nonempty for arr in lst), default=0)

    def to_2d(lst):
        if not lst:
            return np.empty((0, min_length))
        return np.array([a[:min_length] for a in lst])

    residual_errors      = to_2d(residual_errors)
    weighted_mse_errors  = to_2d(weighted_mse_errors)
    train_loss           = to_2d(train_loss)
    val_loss             = to_2d(val_loss)
    res1_pv_sample       = to_2d(res1_pv_sample)
    res2_geowind_sample  = to_2d(res2_geowind_sample)
    val_mse_var_2d       = [to_2d(val_mse_var[i]) for i in range(5)]

    # ------------------- Smooth over time -------------------
    residual_errors      = moving_average_2d(residual_errors, smooth_window)
    weighted_mse_errors  = moving_average_2d(weighted_mse_errors, smooth_window)
    train_loss           = moving_average_2d(train_loss, smooth_window)
    val_loss             = moving_average_2d(val_loss, smooth_window)
    res1_pv_sample       = moving_average_2d(res1_pv_sample, smooth_window)
    res2_geowind_sample  = moving_average_2d(res2_geowind_sample, smooth_window)
    val_mse_var_2d       = [moving_average_2d(x, smooth_window) for x in val_mse_var_2d]

    # ------------------- Means + 95% CI -------------------
    def mean_ci(x2d):
        """np.ndarray[fold, time] -> (mean, low, high) with 95% CI"""
        if x2d.size == 0:
            # keep shapes sensible
            return np.array([]), np.array([]), np.array([])
        n = x2d.shape[0]
        mean = x2d.mean(axis=0)
        std = x2d.std(axis=0)
        half = 1.96 * std / np.sqrt(n) if n else np.zeros_like(mean)
        return mean, mean - half, mean + half

    res_mean, res_low, res_high = mean_ci(residual_errors)
    mse_mean, mse_low, mse_high = mean_ci(weighted_mse_errors)

    train_loss_mean, train_loss_low, train_loss_high = mean_ci(train_loss)
    val_loss_mean,   val_loss_low,   val_loss_high   = mean_ci(val_loss)

    res1_mean, res1_low, res1_high = mean_ci(res1_pv_sample)
    res2_mean, res2_low, res2_high = mean_ci(res2_geowind_sample)

    total_era_res_mean = res1_mean + res2_mean
    total_era_res_low  = res1_low  + res2_low
    total_era_res_high = res1_high + res2_high

    val_mse_var_mean = []
    val_mse_var_low = []
    val_mse_var_high = []
    for i in range(5):
        m, lo, hi = mean_ci(val_mse_var_2d[i])
        val_mse_var_mean.append(m)
        val_mse_var_low.append(lo)
        val_mse_var_high.append(hi)

    # Number of *smoothed* points (time axis length)
    epochs = len(res_mean)

    return {
        "epochs": epochs,

        "res_mean": res_mean, "res_low": res_low, "res_high": res_high,
        "mse_mean": mse_mean, "mse_low": mse_low, "mse_high": mse_high,

        "train_loss_mean": train_loss_mean,
        "train_loss_low": train_loss_low,
        "train_loss_high": train_loss_high,

        "val_loss_mean": val_loss_mean,
        "val_loss_low": val_loss_low,
        "val_loss_high": val_loss_high,

        # sampled residuals + totals
        "res1_mean": res1_mean, "res1_low": res1_low, "res1_high": res1_high,
        "res2_mean": res2_mean, "res2_low": res2_low, "res2_high": res2_high,
        "total_era_res_mean": total_era_res_mean,
        "total_era_res_low": total_era_res_low,
        "total_era_res_high": total_era_res_high,

        # --- NEW: per-variable val_mse_var stats (lists of 5 arrays) ---
        "val_mse_var_mean": val_mse_var_mean,
        "val_mse_var_low": val_mse_var_low,
        "val_mse_var_high": val_mse_var_high,
    }

def fill_in_ax(ax, x, stats, error_type, epochs, color, name, num=None, alpha=0.3, linestyle='-'):
    # pick series (optionally indexed by num)
    if num is None:
        mean = stats[f"{error_type}_mean"]
        low  = stats[f"{error_type}_low"]
        high = stats[f"{error_type}_high"]
    else:
        mean = stats[f"{error_type}_mean"][num]
        low  = stats[f"{error_type}_low"][num]
        high = stats[f"{error_type}_high"][num]

    x_ = x[:epochs]
    mean_ = mean[:epochs]
    low_ = low[:epochs]
    high_ = high[:epochs]

    ax.plot(
        x_,
        mean_,
        label=f"{name} mean",
        linewidth=2.2,
        color=color,
        linestyle=linestyle
    )
    ax.fill_between(x_, low_, high_, alpha=alpha, color=color)

    return ax

def plot_cv_individual_val_metrics(
    model_ids, fold_num, log_path, out_dir,
    smooth_window=10, data_type="era5", prefix=''
):
    """
    Plot cross-validated validation metrics for one or multiple models.
    Saves 5 separate plots (one per variable) into a subfolder.
    """
    # --- Global styling tweaks ---
    plt.rcParams.update({
        "figure.figsize": (13, 5),
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    })

    model_id_to_name = {
        "c1e1": r"c=1e-1",
        "c1e2": r"c=1e-2",
        "c1e3": r"c=1e-3",
        "c1e2_pv": r"$\mathcal{R}_1$: c=1e-2",
        "c1e2_gw": r"$\mathcal{R}_2$: c=1e-2",
    }

    variables_names = [
        r"Eastward Wind, $u$",
        r"Northward Wind, $v$",
        r"Potential Vorticity, $q_{E}$",
        r"Temperature, $T$",
        r"Geopotential, $\Phi$",
    ]
    variables_mse = [
        r"u_{0} - \hat{u}",
        r"v_{0} - \hat{v}",
        r"q_{E,0} - \hat{q}_E",
        r"T_{0} - \hat{T}",
        r"\Phi_{0} - \hat{\Phi}"
    ]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Make a folder for these plots
    models_tag = "_vs_".join(model_ids) if len(model_ids) > 1 else model_ids[0]
    plots_dir = out_dir / f"{models_tag}_individual_val_metrics"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load stats
    stats1 = load_model_stats(
        model_ids[0], smooth_window, fold_num=fold_num,
        log_path=log_path, data_type=data_type
    )

    stats2 = []
    if len(model_ids) > 1:
        stats2 = [
            load_model_stats(
                model_ids[k], smooth_window, fold_num=fold_num,
                log_path=log_path, data_type=data_type
            )
            for k in range(1, len(model_ids))
        ]
    # In case of model with retraining, remove from name after having loaded the data from all retraining runs.
    model_ids=[remove_retrain_suffix(model_id) for model_id in model_ids]

    epochs = min([s["epochs"] for s in [stats1] + stats2]) if stats2 else stats1["epochs"]
    x = np.arange(epochs)

    # 5 separate plots
    for var_i in range(5):
        fig, ax = plt.subplots(figsize=(4.1,4.1))

        fill_in_ax(ax, x, stats1, "val_mse_var", epochs, diffusion_colors[0], "DDPM", num=var_i)

        if stats2:
            for m_i, model_id_2 in enumerate(model_ids[1:]):
                model_name = model_id_2.split("-")[-1].removeprefix(prefix)
                suffix = model_id_to_name.get(model_name, "")
                fill_in_ax(
                    ax, x, stats2[m_i], "val_mse_var", epochs,
                    pidm_colors[model_name],
                    f"PIDM-{suffix}",
                    num=var_i
                )

        xlabel = f"Epoch" + (f" (smoothed, window={smooth_window})" if smooth_window > 1 else "")
        ax.set_xlabel(xlabel)
        ylabel = rf"$\mathbb{{E}}_{{t,\mathbf{{x}}_0}}[\lambda_t \| {variables_mse[var_i]} \|^2]$"
        ax.set_ylabel(ylabel)
        ax.set_title(f"Validation wMSE: {variables_names[var_i]}")
        ax.set_yscale("log")
        ax.legend(frameon=True, fancybox=True, framealpha=0.9, loc="upper right")

        var_slug = ["u", "v", "qe", "T", "Phi"][var_i]
        save_path = plots_dir / f"{models_tag}_val_wmse_{var_slug}.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {save_path}")

    print(f"All plots saved in: {plots_dir}")

def remove_retrain_suffix(model_id):
    for i in range(5):
        if model_id.split('-')[-1]=="retrain":
            model_id = "-".join(model_id.split('-')[:-1])
    return model_id

def plot_cv_val_metrics(model_ids, fold_num, log_path, out_dir, smooth_window=10, data_type="era5", prefix = ''):
    """
    Function to plot cross-validated validation metrics for one or multiple models.
    Args:
        model_ids (list): List of model IDs to plot.
        fold_num (int): Number of cross-validation folds.
        log_path (str): Path to the logs directory.
        out_dir (str): Output directory to save the plots.
        smooth_window (int): Window size for smoothing the metrics.
    """
    # --- Global styling tweaks ---
    plt.rcParams.update({
        "figure.figsize": (13, 5),
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    })

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats1 = load_model_stats(model_ids[0], smooth_window, fold_num=fold_num, log_path=log_path, data_type=data_type)
    if len(model_ids) > 1:
        stats2 = [load_model_stats(model_ids[i], smooth_window, fold_num=fold_num, log_path=log_path, data_type=data_type) for i in range(1, len(model_ids))]
    # In case of model with retraining, remove from name after having loaded the data from all retraining runs.
    model_ids=[remove_retrain_suffix(model_id) for model_id in model_ids]

    epochs = min([stats["epochs"] for stats in [stats1]+stats2]) if len(model_ids) > 1 else stats1["epochs"]

    x = np.arange(epochs)

    fig, axes = plt.subplots(1, 3, sharex=True)

    # ---------------- Residual ----------------
    ax = axes[0]

    # fill in the loss curves for diffusion and pidm
    ax = fill_in_ax(ax, x, stats1, 'res', epochs, diffusion_colors[0], "DDPM")
    if len(model_ids) > 1:
        for i, model_id_2 in enumerate(model_ids[1:]):
            model_name = model_id_2.split('-')[-1].removeprefix(prefix)
            suffix = model_id_to_name.get(model_name, "")
            ax = fill_in_ax(ax, x, stats2[i], 'res', epochs, pidm_colors[model_name], f"PIDM-{suffix}")

    ax.set_xlabel(f"Epoch{f" (smoothed, window={smooth_window})" if smooth_window>1 else ""}")
    ax.set_ylabel(r"$\mathcal{R}_1(\hat{x}_0) + \mathcal{R}_2(\hat{x}_0)$")
    # ax.set_ylabel(r"$\mathcal{R}_{\text{MAE}}(\mathbf{x_0}) \sim p_\theta (\mathbf{x_0})$")
    ax.set_title(r"MAE of residuals through mean estimation over validation set")
    ax.set_yscale("log")
    ax.legend(frameon=True, fancybox=True, framealpha=0.9, loc="upper right")

    # ---------------- MSE ----------------
    ax = axes[1]

    # fill in the loss curves for diffusion and pidm
    ax = fill_in_ax(ax, x, stats1,"mse", epochs, diffusion_colors[0], "DDPM")
    if len(model_ids) > 1:
        for i, model_id_2 in enumerate(model_ids[1:]):
            model_name = model_id_2.split('-')[-1].removeprefix(prefix)
            suffix = model_id_to_name.get(model_name, "")
            ax = fill_in_ax(ax, x, stats2[i], 'mse', epochs, pidm_colors[model_name], f"PIDM-{suffix}")

    ax.set_xlabel(f"Epoch{f" (smoothed, window={smooth_window})" if smooth_window>1 else ""}")
    ax.set_ylabel(r"$\mathbb{E}_{t, \mathbf{x_0}}[\lambda_t \|\mathbf{x_0} - \hat{\mathbf{x}}_0 (\mathbf{x}_t,t) \|^2]$")
    ax.set_title("Validation Weighted MSE")
    ax.set_yscale("log")
    ax.legend(frameon=True, fancybox=True, framealpha=0.9, loc="upper right")

    ax = axes[2]

    # Train loss
    ax = fill_in_ax(ax, x, stats1,"train_loss", epochs, train_loss_diff_colors[0], "DDPM Train loss", linestyle='--')
    ax = fill_in_ax(ax, x, stats1,"val_loss", epochs, val_loss_diff_colors[0], "DDPM Val loss")
    if len(model_ids) > 1:
        for i, model_id_2 in enumerate(model_ids[1:]):
            model_name = model_id_2.split('-')[-1].removeprefix(prefix)
            suffix = model_id_to_name.get(model_name, "")
            ax = fill_in_ax(ax, x, stats2[i],"train_loss", epochs, train_loss_pidm_colors[model_name], f"PIDM-{suffix} Train loss", linestyle='--')
            ax = fill_in_ax(ax, x, stats2[i],"val_loss", epochs, val_loss_pidm_colors[model_name], f"PIDM-{suffix} Val loss")

    ax.set_xlabel(f"Epoch{f" (smoothed, window={smooth_window})" if smooth_window>1 else ""}")
    ax.set_ylabel(
        r"$\mathbb{E}_{t, \mathbf{x_0}}[\lambda_t \|\mathbf{x_0} - \hat{\mathbf{x}}_0 (\mathbf{x}_t,t) \|^2] + "
        r"\frac{1}{2 \tilde{\Sigma}} || \mathcal{R}(\mathbf{x}_0^*(\mathbf{x}_t,t))||^2$"
    )
    ax.set_title("Train vs Validation loss")
    ax.set_yscale("log")
    ax.legend(frameon=True, fancybox=True, framealpha=0.9, loc="upper right")

    # Improve spacing
    fig.tight_layout(rect=[0, 0.0, 1, 0.95])

    save_path = out_dir / f"{"_vs_".join(model_ids)}_val_metrics.png" if len(model_ids) > 1 else out_dir / f"{model_ids[0]}_val_metrics.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved validation metrics plot to {save_path}")


def plot_models_train_and_mse(model_ids, log_path="logs", out_dir=Path("./reports/figures/model_comparison"), log_scale=True):
    """
    Plot training loss and weighted MSE for a list of models (each model should have a
    `logs/<model_id>/version_0/metrics.csv` file with at least `epoch`, `train_loss`, and `train_mse_(weighted)` columns).

    Produces a two-panel figure: left = training loss vs epoch, right = train weighted MSE vs epoch.
    Colors follow the existing ERA5 conventions: use `train_loss_pidm_colors` for training loss when available
    and `pidm_colors` for MSE lines. Falls back to the matplotlib cycle when a mapping is not found.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # helper to read and aggregate per-epoch mean
    def load_epoch_stats(model_id):
        prev_run_exists = False
        if "retrain" in model_id:
            # if retrain search for prev version, should be remove fold nr and "retrain"
            prev_mid = "-".join(model_id.split("-")[:-1])
            prev_df = load_epoch_stats(prev_mid)
            prev_run_exists = True
        csv_path = Path(log_path) / model_id / "version_0" / "metrics.csv"
        df = pd.read_csv(csv_path).apply(pd.to_numeric, errors="coerce").dropna(subset=["step"]).sort_values("step")

        if prev_run_exists:
            last_epoch = prev_df["epoch"].max()
            df["epoch"]+=last_epoch+1
            return pd.concat([prev_df, df])
        return df

    # prepare figure
    fig, ax_loss = plt.subplots(figsize=(4, 4))
    stats = [load_epoch_stats(model_id) for model_id in model_ids]

    ax_loss.plot(stats[0]["epoch"], stats[0]["train_loss"].values, label="DDPM Loss/wMSE", color=train_loss_diff_colors[0])
    ax_loss.plot(stats[0]["epoch"], stats[1]["train_loss"].values, label="PIDM-"+model_id_to_name.get("c1e2_pv", "")+" Loss", color=train_loss_pidm_colors["c1e2_pv"])
    ax_loss.plot(stats[0]["epoch"], stats[1]["train_mse_(weighted)"].values, label="PIDM-"+model_id_to_name.get("c1e2_pv", "")+" wMSE", color="#D8979C", linestyle='--')

    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Train Loss")
    ax_loss.set_title("Training Loss per Epoch")
    ax_loss.set_yscale("log")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend()

    save_path = out_dir / f"{'_vs_'.join(model_ids)}_train_loss{PLOT_TYPE}"
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison plot to {save_path}")

def plot_and_save_era5(csv_path, out_dir, loss_title="Loss", residual_title="Residuals & MSE", log_scale=False):
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path).apply(pd.to_numeric).dropna(subset=["step"])
    df = df.sort_values("step")

    # ---- Loss plot ----
    train_df = df.dropna(subset=["train_loss"])
    val_df = df.dropna(subset=["val_loss"])

    plt.plot(train_df.epoch, train_df.train_loss, label="train_loss")
    plt.plot(val_df.epoch, val_df.val_loss, label="val_loss")
    plt.legend(); plt.grid(); plt.title("Loss")
    plt.savefig(out_dir / "loss.png"); plt.clf()

    res_df_geo = df.dropna(subset=["val_era5_geo_wind_residual(norm)"])
    res_df_planetary = df.dropna(subset=["val_era5_planetary_residual(norm)"])
    mse_df = df.dropna(subset=["val_mse_(weighted)"])

    # ---- Residual + MSE plot ----
    res_df_geo = df.dropna(subset=["val_era5_geo_wind_residual(norm)"])
    res_df_planetary = df.dropna(subset=["val_era5_planetary_residual(norm)"])
    mse_df = df.dropna(subset=["val_mse_(weighted)"])

    plt.plot(res_df_geo.epoch, res_df_geo["val_era5_geo_wind_residual(norm)"], label="geo_wind_residual (normalized)")
    plt.plot(res_df_planetary.epoch, res_df_planetary["val_era5_planetary_residual(norm)"], label="planetary_residual (normalized)")
    plt.plot(mse_df.epoch, mse_df["val_mse_(weighted)"], label="mse_weighted")

    if log_scale:
        plt.yscale("log")

    plt.legend()
    plt.grid(True, which="both")
    plt.title(residual_title)
    plt.savefig(out_dir / "residuals.png", bbox_inches="tight")
    plt.clf()
    print(f"saved fig to {out_dir}/residuals.png")


def era5_models_summary_latex_table(
    model_ids,
    fold_num,
    log_path,
    smooth_window=10,
    caption="ERA5 comparison of metrics of mean across folds",
    label="tab:era5_summary",
    sig_figs=3,
    wrap_math=True,
    resize_to_linewidth=True,
    model_id_to_header=None,
):
    """
    Build a LaTeX table summarizing ERA5 validation metrics across folds,
    using load_model_stats() outputs (mean across folds + smoothing).

    Rows:
      - Total residual @ best val loss
      - R1 @ best val loss
      - R2 @ best val loss
      - Weighted MSE @ best val loss
      - Weighted MSE (best)
      - Total residual @ best Weighted MSE
    """
    import math
    import numpy as np

    ROWS = [
        ("total_at_vloss", r"Total residual $(\mathcal{R}_1+\mathcal{R}_2)$ @ best val loss", "min"),
        ("r1_at_vloss",    r"$\mathcal{R}_1$ (planetary) @ best val loss",                    "min"),
        ("r2_at_vloss",    r"$\mathcal{R}_2$ (geo wind) @ best val loss",                     "min"),
        ("wmse_at_vloss",  r"Weighted MSE @ best val loss",                                   "min"),
        ("wmse_best",      r"Weighted MSE (best)",                                            "min"),
        ("total_at_wmse",  r"Total residual @ best Weighted MSE",                             "min"),
    ]

    # ---------- formatting helpers ----------
    def to_latex_sci(x: float, decimals: int = 2) -> str:
        if x == 0:
            return f"{0:.{decimals}f}" if decimals > 0 else "0"
        if not math.isfinite(x):
            return r"\infty" if x > 0 else (r"-\infty" if x < 0 else r"\mathrm{nan}")

        sign = "-" if x < 0 else ""
        ax = abs(x)
        exp = int(math.floor(math.log10(ax)))
        mant = ax / (10 ** exp)

        mant_rounded = round(mant, decimals)
        if mant_rounded >= 10:
            mant_rounded /= 10
            exp += 1

        mant_str = f"{mant_rounded:.{decimals}f}"
        if exp == 0:
            return f"{sign}{mant_str}"
        if float(mant_str) == 1.0:
            return f"{sign}10^{{{exp}}}"
        return f"{sign}{mant_str}\\times 10^{{{exp}}}"

    def tex_escape(s: str) -> str:
        return s.replace("_", r"\_")

    def fmt_cell(val: float, style: str = None) -> str:
        core = to_latex_sci(float(val), decimals=2)
        if style == "best":
            core = r"\boldsymbol{" + core + "}"
        elif style == "second":
            core = r"\underline{" + core + "}"
        return f"${core}$" if wrap_math else core

    # ---------- load per-model stats via your helper ----------
    # Expecting load_model_stats to be available in scope.
    model_stats = {}
    for mid in model_ids:
        model_stats[mid] = load_model_stats(
            mid,
            smooth_window=smooth_window,
            fold_num=fold_num,
            log_path=log_path,
            data_type="era5",
        )

    # Align all models to common length
    common_T = min(model_stats[mid]["epochs"] for mid in model_ids)

    # ---------- compute values ----------
    values = {k: {} for (k, _, _) in ROWS}

    for mid in model_ids:
        s = model_stats[mid]

        # mean series (already smoothed + mean across folds)
        # NOTE: use the per-component residual means (res1/res2) so total = res1+res2,
        # rather than s["total_era_res_mean"] (either works; this is explicit).
        r1    = np.asarray(s["res1_mean"][:common_T], dtype=float)
        r2    = np.asarray(s["res2_mean"][:common_T], dtype=float)
        total = r1 + r2

        wmse  = np.asarray(s["mse_mean"][:common_T], dtype=float)
        vloss = np.asarray(s["val_loss_mean"][:common_T], dtype=float)

        # "best" epoch := lowest validation loss
        i_vloss = int(np.argmin(vloss))
        # absolute best wMSE epoch
        i_wmse = int(np.argmin(wmse))

        values["total_at_vloss"][mid] = {"val": float(total[i_vloss]), "epoch": i_vloss}
        values["r1_at_vloss"][mid]    = {"val": float(r1[i_vloss]),    "epoch": i_vloss}
        values["r2_at_vloss"][mid]    = {"val": float(r2[i_vloss]),    "epoch": i_vloss}
        values["wmse_at_vloss"][mid]  = {"val": float(wmse[i_vloss]),  "epoch": i_vloss}

        values["wmse_best"][mid]      = {"val": float(wmse[i_wmse]),   "epoch": i_wmse}
        values["total_at_wmse"][mid]  = {"val": float(total[i_wmse]),  "epoch": i_wmse}

    # ---------- determine best + second-best per row ----------
    best_and_second = {}
    for (row_key, _, direction) in ROWS:
        items = [(mid, values[row_key][mid]["val"]) for mid in model_ids]
        items_sorted = sorted(items, key=lambda x: x[1], reverse=(direction == "max"))
        best_mid = items_sorted[0][0] if len(items_sorted) > 0 else None
        second_mid = items_sorted[1][0] if len(items_sorted) > 1 else None
        best_and_second[row_key] = (best_mid, second_mid)

    # ---------- header names ----------
    if model_id_to_header is None:
        headers = {mid: tex_escape(mid) for mid in model_ids}
    else:
        headers = {mid: model_id_to_header.get(mid, tex_escape(mid)) for mid in model_ids}

    # ---------- build LaTeX ----------
    colspec = "l" + "c" * len(model_ids)

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + caption + r"}")
    lines.append(r"\label{" + label + r"}")

    if resize_to_linewidth:
        lines.append(r"\resizebox{\linewidth}{!}{%")

    lines.append(r"\begin{tabular}{" + colspec + r"}")
    lines.append(r"\toprule")
    lines.append("Metric & " + " & ".join(headers[mid] for mid in model_ids) + r" \\")
    lines.append(r"\midrule")

    for (row_key, row_label, _) in ROWS:
        best_mid, second_mid = best_and_second[row_key]
        row_cells = [row_label]
        for mid in model_ids:
            style = None
            if mid == best_mid:
                style = "best"
            elif mid == second_mid:
                style = "second"
            row_cells.append(fmt_cell(values[row_key][mid]["val"], style=style))
        lines.append(" & ".join(row_cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    if resize_to_linewidth:
        lines.append(r"}")  # closes resizebox

    lines.append(r"\end{table}")

    return "\n".join(lines)




def plot_cv_residual_metrics_era5(model_ids, fold_num, log_path, out_dir, smooth_window=10, data_type="era5", prefix = ''):
    """
    Function to plot cross-validated validation metrics for one or multiple models.
    Args:
        model_ids (list): List of model IDs to plot.
        fold_num (int): Number of cross-validation folds.
        log_path (str): Path to the logs directory.
        out_dir (str): Output directory to save the plots.
        smooth_window (int): Window size for smoothing the metrics.
    """
    # --- Global styling tweaks ---
    plt.rcParams.update({
        "figure.figsize": (13, 5),
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    })

    model_id_to_name={
        "c1e1": r"c=1e-1",
        "c1e2": r"c=1e-2",
        "c1e3": r"c=1e-3",
        "c1e2_pv": r"$\mathcal{R}_1$: c=1e-2",
        "c1e2_gw": r"$\mathcal{R}_2$: c=1e-2",
    }    

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats1 = load_model_stats(model_ids[0], smooth_window, fold_num=fold_num, log_path=log_path)
    if len(model_ids) > 1:
        stats2 = [load_model_stats(model_ids[i], smooth_window, fold_num=fold_num, log_path=log_path) for i in range(1, len(model_ids))]
    # In case of model with retraining, remove from name after having loaded the data from all retraining runs.
    model_ids=[remove_retrain_suffix(model_id) for model_id in model_ids]
    
    epochs = min([stats["epochs"] for stats in [stats1]+stats2]) if len(model_ids) > 1 else stats1["epochs"]

    x = np.arange(epochs)

    fig, axes = plt.subplots(1, 3, sharex=True)

    # ---------------- Residual ----------------
    ax = axes[0]

    res_str = "total_era_res" if data_type == "era5" else "res"
    ax = fill_in_ax(ax, x, stats1, res_str, epochs, diffusion_colors[0], "DDPM")
    if len(model_ids) > 1:
        for i, model_id_2 in enumerate(model_ids[1:]):
            model_name = model_id_2.split('-')[-1].removeprefix(prefix)
            suffix = model_id_to_name.get(model_name, "")
            ax = fill_in_ax(ax, x, stats2[i], res_str, epochs, pidm_colors[model_name], f"PIDM-{suffix}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$\mathcal{R}_{\text{MAE}}(\mathbf{x_0}) \sim p_\theta (\mathbf{x_0})$")
    ax.set_title("Validation Sample Residual")
    ax.set_yscale("log")
    ax.legend(frameon=True, fancybox=True, framealpha=0.9, loc="upper right")
    # ---------------- MSE ----------------
    ax = axes[1]

    # fill in the loss curves for diffusion and pidm
    ax = fill_in_ax(ax, x, stats1,"res1", epochs, diffusion_colors[0], "DDPM")
    if len(model_ids) > 1:
        for i, model_id_2 in enumerate(model_ids[1:]):
            model_name = model_id_2.split('-')[-1].removeprefix(prefix)
            suffix = model_id_to_name.get(model_name, "")
            ax = fill_in_ax(ax, x, stats2[i],"res1", epochs, pidm_colors[model_name], f"PIDM-{suffix}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(r"$\mathcal{R1}_{\text{MAE}}(\mathbf{x_0}) \sim p_\theta (\mathbf{x_0})$")
        ax.set_title("Planetary Vorticity Sample Residual")
        ax.set_yscale("log")
        ax.legend(frameon=True, fancybox=True, framealpha=0.9, loc="upper right")

    ax = axes[2]

    # Train loss
    ax = fill_in_ax(ax, x, stats1,"res2", epochs, diffusion_colors[0], "DDPM")
    if len(model_ids) > 1:
        for i, model_id_2 in enumerate(model_ids[1:]):
            model_name = model_id_2.split('-')[-1].removeprefix(prefix)
            suffix = model_id_to_name.get(model_name, "")
            ax = fill_in_ax(ax, x, stats2[i],"res2", epochs, pidm_colors[model_name], f"PIDM-{suffix}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$\mathcal{R2}_{\text{MAE}}(\mathbf{x_0}) \sim p_\theta (\mathbf{x_0})$")
    ax.set_title("Geostrophic Wind Sample Residual")
    ax.set_yscale("log")
    ax.legend(frameon=True, fancybox=True, framealpha=0.9)

    # Improve spacing
    fig.tight_layout(rect=[0, 0.0, 1, 0.95])

    save_path = out_dir / f"{"_vs_".join(model_ids)}_residual_metrics.png" if len(model_ids) > 1 else out_dir / f"{model_ids[0]}_residual_metrics.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved validation metrics plot to {save_path}")

def era5_residuals_plot(model, conditional, model_id, normalize=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conditional = conditional.to(device)
    model = model.to(device)
    loss_fn = model.loss_fn
    loss_fn.set_mean_and_std(model.means, model.stds, model.diff_means, model.diff_stds)
    model_sample = model.sample_loop(conditionals = conditional)
    num_vars = (conditional.shape[1] - 8) // 6
    loss_geo_wind = loss_fn.compute_residual_geostrophic_wind(x0_previous=conditional[:, num_vars*3+4:-4], x0_change_pred=model_sample, normalize=normalize).abs().mean(1)
    loss_planetary = loss_fn.compute_residual_planetary_vorticity(x0_previous=conditional[:, num_vars*3+4:-4], x0_change_pred=model_sample, normalize=normalize).abs().mean(1)
    print(f"mean loss geo wind: {loss_geo_wind.mean().item()}")
    print(f"mean loss planetary vorticity: {loss_planetary.mean().item()}")
    residual_variables = [
        'qgpv_R1',
        'planetary_vorticity_R2',
        'geo_wind_R3'
    ]
    residual_losses = [loss_qgpv,loss_planetary,loss_geo_wind]

    for res_var, res_loss in zip(residual_variables, residual_losses):
        residual = res_loss[0].cpu().numpy()
        visualize_era5_sample(residual, res_var, "500", big_data_sample=None, dir=Path("./reports/figures") / model_id / f"residuals")


if __name__ == "__main__":
    from pde_diff.utils import DatasetRegistry, LossRegistry
    plot_darcy = False
    plot_data_samples = True
    plot_era5_training = False
    plot_era5_residual = False
    plot_era5_residual_metrics = False
    plot_era5_individual_var_mse = False
    era5_latex = False
    plot_darcy_sample = False

    model_path = Path('./models')
    model_ids = ['era5_clean_hp3-baseline-retrain-retrain','era5_clean_hp3-c1e2_pv-retrain-retrain']

    # Quick comparison plot for simple training logs (epoch, step, train_loss, train_mse_(weighted))
    plot_era5_long_compare_models = False
    
    if plot_era5_long_compare_models:
        compare_model_ids = ['era5_clean_hp3-baseline-full-retrain-retrain','era5_clean_hp3-c1e2_pv-full-retrain-retrain']

        plot_models_train_and_mse(compare_model_ids, log_path="logs", out_dir=Path(f"reports/figures/compare_models"), log_scale=False)

    if plot_era5_training:
        # PLOT ERA 5 LOSS:
        # -------------------------------
        plot_cv_val_metrics(
            model_ids=model_ids,
            fold_num=5,
            log_path="logs",
            out_dir=f"reports/figures/era5_baseline_comparisons",
            smooth_window=1,
            data_type="era5",
            prefix=""
        )
        # ---------------------------------------------------

    if plot_era5_residual:
        #Load model
        model_id = 'exp1-ghxhv'
        cfg = OmegaConf.load(model_path / (model_id) / "config.yaml")
        diffusion_model = DiffusionModel(cfg)
        diffusion_model.load_model(model_path / model_id / f"best-val_loss-weights.pt")

        #Load data
        dataset = DatasetRegistry.create(cfg.dataset)
        dataset_train, dataset_val = split_dataset(cfg, dataset)

        conditional = dataset_val[0][0]
        conditional = torch.tensor(conditional).unsqueeze(0).to('cpu')

        era5_residuals_plot(diffusion_model, conditional, model_id, normalize=False)
        breakpoint()

    if plot_era5_individual_var_mse:

        plot_cv_individual_val_metrics(
            model_ids=model_ids,
            fold_num=5,
            log_path="logs",
            out_dir=f"reports/figures/era5_baseline_comparisons",
            smooth_window=1,
            prefix = 'ne_'
        )
    if plot_era5_residual_metrics:

        plot_cv_residual_metrics_era5(
            model_ids=model_ids,
            fold_num=5,
            log_path="logs",
            out_dir=f"reports/figures/era5_baseline_comparisons",
            smooth_window=1,
            prefix = 'ne_'
        )

    if plot_darcy:
        model_path = Path('./models')
        model_id = 'era5_cleanhp_50e-bqlmk'
        model_id_2 = 'era5_cleanhp_50e-hgrnf'

        plot_darcy_val_metrics(
            model_id_1=model_id,
            model_id_2=model_id_2,
            fold_num=5,
            log_path="logs",
            out_dir=f"reports/figures/{model_id}",
            smooth_window=20,
        )

        cfg = OmegaConf.load(model_path / model_id / "config.yaml")
        dataset = DatasetRegistry.create(cfg.dataset)
        diffusion_model = DiffusionModel(cfg)
        diffusion_model.load_model(model_path / model_id / f"best-val_loss-weights.pt")
        diffusion_model = diffusion_model.to('cuda' if torch.cuda.is_available() else 'cpu')

    if plot_darcy_sample:
        model_path = Path('./models')
        model_id_1 = 'exp1-aaaaa-1'
        cfg_1 = OmegaConf.load(model_path / model_id_1 / "config.yaml")
        model_id_2 = 'exp1-dp'
        cfg_2 = OmegaConf.load(model_path / model_id_2 / "config.yaml")
        diffusion_model_1 = DiffusionModel(cfg_1)
        diffusion_model_1.load_model(model_path / model_id_1 / f"best-val_loss-weights.pt")
        diffusion_model_2 = DiffusionModel(cfg_2)
        diffusion_model_2.load_model(model_path / model_id_2 / f"best-val_loss-weights.pt")
        plot_darcy_samples(diffusion_model_1, diffusion_model_2, model_id_2, Path('./reports/figures') / model_id_2)

    if era5_latex:
        pretty = {
            "era5_clean_hp3-mbaseline": r"Baseline",
            "era5_clean_hp3-ne_c1e1": r"$c=10^{-1}$",
            "era5_clean_hp3-ne_c1e2": r"$c=10^{-2}$",
            "era5_clean_hp3-ne_c1e3": r"$c=10^{-3}$",
            "era5_clean_hp3-c1e2_pv": r"$\mathcal{R}_1$: $c=10^{-2}$",
            "era5_clean_hp3-c1e2_gw": r"$\mathcal{R}_2$: $c=10^{-2}$",
            'era5_clean_hp3-baseline-retrain-retrain': r"Baseline (Extended training)",
            'era5_clean_hp3-c1e2_pv-retrain-retrain': r"$\mathcal{R}_1$: $c=10^{-2}$ (Extended training)"
        }

        latex = era5_models_summary_latex_table(
            model_ids=model_ids,
            fold_num=5,
            log_path="logs",
            smooth_window=1,
            model_id_to_header=pretty,
        )
        print(latex)


    if plot_data_samples:
        # Load the dataset configuration
        config_path = Path("configs/dataset/era5.yaml")
        cfg = OmegaConf.load(config_path)
        cfg.normalize = False

        # Initialize the ERA5 dataset
        era5_dataset = ERA5Dataset(cfg)
        config_path = Path("configs/dataset/era5_full.yaml")
        cfg = OmegaConf.load(config_path)
        cfg.normalize = False
        cfg.lat_range = None
        #era5_dataset_full = ERA5Dataset(cfg)

        # Visualize all variables of a sample from the dataset
        sample_idx = 0  # Index of the sample to visualize
        variable = "t"  # Variable to visualize
        level = 500  # Pressure level in hPa
        #for variable in cfg.atmospheric_features:
        #    data_sample = get_data_sample(era5_dataset, sample_idx, variable, level)
        #    data_sample_full = get_data_sample(era5_dataset_full, sample_idx, variable, level)
        #    visualize_era5_sample(data_sample, variable, level, big_data_sample=data_sample_full)

        # Visualize the noise schedule
        config_path = Path("configs/scheduler/ddpm.yaml")
        cfg_scheduler = OmegaConf.load(config_path)
        from pde_diff.utils import SchedulerRegistry
        scheduler = SchedulerRegistry.create(cfg_scheduler)

        variable = "t"
        #data_sample = get_data_sample(era5_dataset, sample_idx, variable, level)
        _,change = era5_dataset[0]

        visualize_noise_schedule(scheduler, change[10], variable, level, sample_idx)

        # Visualize a full sample from the dataset without the subset overlay
        #visualize_era5_sample_full(data_sample_full, variable, level)

        # Visualize time series
        #for variable in cfg.atmospheric_features:
        #    visualize_time_series(era5_dataset, variable=variable, level=500, coords=(12.568, 55.676))
