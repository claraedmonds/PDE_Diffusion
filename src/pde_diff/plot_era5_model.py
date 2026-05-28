"""
Generate diagnostic plots for a single trained ERA5 model.

Usage:
    python src/pde_diff/plot_era5_model.py --model-id tiny-pycmm

Outputs to reports/figures/<model_id>/:
    loss.png          -- train / val loss curves
    residuals.png     -- vorticity residuals + weighted MSE (if available)
    sample_<var>_<level>hPa.png  -- one field per atmospheric variable
    noise_schedule.png -- diffusion noise corruption of a sample field
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed on cluster
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

from pde_diff.visualize import (
    get_data_sample,
    plot_and_save_era5,
    visualize_era5_sample,
    visualize_noise_schedule,
    VAR_NAMES,
)
from pde_diff.data.datasets import ERA5Dataset
from pde_diff.utils import DatasetRegistry, SchedulerRegistry


PRESSURE_LEVEL = 500  # hPa to visualise


def find_log_csv(model_id: str, log_root: Path) -> Path | None:
    for version in ["version_0", "version_1", "version_2"]:
        p = log_root / model_id / version / "metrics.csv"
        if p.exists():
            return p
    return None


def plot_samples(dataset: ERA5Dataset, out_dir: Path, n_samples: int = 1) -> None:
    for variable in dataset.atmospheric_features:
        for sample_idx in range(n_samples):
            data = get_data_sample(dataset, sample_idx, variable, level=PRESSURE_LEVEL)
            visualize_era5_sample(
                data,
                variable,
                level=PRESSURE_LEVEL,
                sample_idx=sample_idx,
                dir=out_dir,
            )


def plot_noise_schedule(dataset: ERA5Dataset, cfg, out_dir: Path) -> None:
    cfg_scheduler = OmegaConf.load("configs/scheduler/ddpm.yaml")
    scheduler = SchedulerRegistry.create(cfg_scheduler)
    _, state_change = dataset[0]
    # pick the temperature channel (index 9 = t * 3 levels, level 0)
    t_channel = dataset.atmospheric_features.index("t") * len(dataset.pressure_levels)
    visualize_noise_schedule(
        scheduler,
        state_change[t_channel],
        variable="t",
        level=PRESSURE_LEVEL,
        sample_idx=0,
        dir=out_dir,
        steps=5,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True, help="Model ID, e.g. tiny-pycmm")
    parser.add_argument("--model-dir", default="models", help="Root directory for saved models")
    parser.add_argument("--log-dir",   default="logs",   help="Root directory for training logs")
    parser.add_argument("--out-dir",   default="reports/figures", help="Output directory for plots")
    parser.add_argument("--no-samples", action="store_true", help="Skip data sample plots (faster)")
    args = parser.parse_args()

    model_dir = Path(args.model_dir) / args.model_id
    out_dir   = Path(args.out_dir)   / args.model_id
    out_dir.mkdir(parents=True, exist_ok=True)

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    cfg_path = model_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.yaml not found in {model_dir}")

    print(f"Loading config from {cfg_path}")
    cfg = OmegaConf.load(cfg_path)

    # --- Loss / residual curves ---
    csv_path = find_log_csv(args.model_id, Path(args.log_dir))
    if csv_path:
        print(f"Plotting metrics from {csv_path}")
        plot_and_save_era5(csv_path, out_dir, log_scale=True)
    else:
        print("Warning: no metrics.csv found, skipping loss plots")

    if args.no_samples:
        print("Skipping sample plots (--no-samples)")
        return

    # --- Data samples ---
    print("Loading dataset for sample plots...")
    dataset_cfg = OmegaConf.create(OmegaConf.to_container(cfg.dataset, resolve=True))
    dataset_cfg.normalize = False  # plot raw physical units
    dataset = ERA5Dataset(dataset_cfg)

    print("Plotting atmospheric variable samples...")
    plot_samples(dataset, out_dir)

    print("Plotting noise schedule...")
    plot_noise_schedule(dataset, cfg, out_dir)

    print(f"\nAll plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
