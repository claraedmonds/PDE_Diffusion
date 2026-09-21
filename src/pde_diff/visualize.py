"""
Pre-training / dataset-exploration visualization helpers (sample slices,
time series at a point, noise-schedule demos). Post-training inference and
evaluation plotting lives in pde_diff.eval.*; shared low-level primitives and
style/constants live in pde_diff.eval_primitives.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

from pde_diff.data.datasets import ERA5Dataset

from pde_diff.eval_primitives import EXTENT_SUBSET, PLOT_TYPE, VAR_NAMES, COLOR_BARS


def get_data_sample(dataset, sample_idx, variable, level=500):
    """Extract data for visualization from the ERA5 dataset."""
    sample = dataset[sample_idx]
    inputs, _ = sample

    var_idx = dataset.atmospheric_features.index(variable)
    level_idx = np.where(dataset.pressure_levels == level)[0][0]

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
    time_series = dataset.data[variable][:, level_idx, lon_idx, lat_idx]
    return time_series


def visualize_noise_schedule(scheduler, data_sample, variable, level=500, sample_idx=None, dir=Path("./reports/figures/samples"), steps=10):
    """Visualize a sample from the ERA5 dataset across diffusion noise steps."""
    fig, ax = plt.subplots(figsize=(data_sample.shape[1] / 5, data_sample.shape[0] / 5))
    ax.imshow(data_sample.T, cmap=COLOR_BARS.get(variable, 'coolwarm'), extent=EXTENT_SUBSET, origin='lower')
    ax.set_xlim(EXTENT_SUBSET[0], EXTENT_SUBSET[1])
    ax.set_ylim(EXTENT_SUBSET[3], EXTENT_SUBSET[2])
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    plot_path = f"era5_sample{sample_idx if sample_idx is not None else ''}_{variable}_{level}hPa_noise_{0}"
    plot_path += PLOT_TYPE
    plot_path = dir / plot_path
    plt.savefig(plot_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"Saved visualization to {plot_path}")

    for t in range(100, scheduler.config.num_train_timesteps + 1, scheduler.config.num_train_timesteps // steps):
        noised_sample = scheduler.add_noise(torch.tensor(data_sample).unsqueeze(0).unsqueeze(0).float(), torch.randn_like(torch.tensor(data_sample).unsqueeze(0).unsqueeze(0).float()), torch.tensor([t - 1]))
        noised_sample = noised_sample.squeeze().squeeze().numpy()
        fig, ax = plt.subplots(figsize=(noised_sample.shape[1] / 5, noised_sample.shape[0] / 5))
        ax.imshow(noised_sample.T, cmap=COLOR_BARS.get(variable, 'coolwarm'), extent=EXTENT_SUBSET, origin='lower')
        ax.set_xlim(EXTENT_SUBSET[0], EXTENT_SUBSET[1])
        ax.set_ylim(EXTENT_SUBSET[3], EXTENT_SUBSET[2])
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

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
    plt.close(fig)
    print(f"Saved time series plot to {plot_path}")


if __name__ == "__main__":
    # Demo: visualize the noise schedule applied to a sample from the dataset.
    config_path = Path("configs/dataset/era5.yaml")
    cfg = OmegaConf.load(config_path)
    cfg.normalize = False

    era5_dataset = ERA5Dataset(cfg)

    sample_idx = 0
    variable = "t"
    level = 500

    config_path = Path("configs/scheduler/ddpm.yaml")
    cfg_scheduler = OmegaConf.load(config_path)
    from pde_diff.utils import SchedulerRegistry
    scheduler = SchedulerRegistry.create(cfg_scheduler)

    _, change = era5_dataset[0]
    visualize_noise_schedule(scheduler, change[10], variable, level, sample_idx)
