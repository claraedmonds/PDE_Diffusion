"""
One-time preprocessing: walk per-year ERA5 zarr stores and materialise every
(input, target) sample as numpy memory-mapped arrays for fast training.

Processes one year at a time to keep peak RAM low (~2 GB/year).
If interrupted, re-running will resume from where it left off (years already
written are skipped based on the progress file).

Usage:
    python src/pde_diff/data/precompute_dataset.py \
        --dataset-config configs/dataset/era5.yaml \
        --zarr-dir      ./data/era5 \
        --years         2015 2016 ... 2024 \
        --out-dir       ./data/era5/precomputed

Output layout:
    <out-dir>/
        inputs.npy      shape (N_total, C_in, H, W), float32, memory-mapped
        targets.npy     shape (N_total, C_out, H, W), float32, memory-mapped
        metadata.json   shapes + normalization stats for PrecomputedERA5Dataset
        progress.json   list of completed years (for resuming)
"""
import argparse
import json
from pathlib import Path
from copy import deepcopy

import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from pde_diff.data.datasets import ERA5Dataset


def samples_for_year(cfg_base, zarr_dir: Path, year: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one year's zarr into RAM and return all (input, target) arrays."""
    cfg = deepcopy(cfg_base)
    cfg.path = str(zarr_dir / f"zarr_{year}")
    cfg.max_year = int(year)

    dataset = ERA5Dataset(cfg)
    N = len(dataset)
    print(f"  {year}: {N} samples — loading into RAM ...", flush=True)
    dataset.data = dataset.data.load()
    print(f"  {year}: in-memory, iterating ...", flush=True)

    sample_input, sample_target = dataset[0]
    inputs = np.empty((N, *sample_input.shape), dtype=np.float32)
    targets = np.empty((N, *sample_target.shape), dtype=np.float32)
    inputs[0] = sample_input
    targets[0] = sample_target
    for i in tqdm(range(1, N), desc=f"  {year}", leave=False):
        inputs[i], targets[i] = dataset[i]

    return inputs, targets


def precompute(dataset_config_path: Path, zarr_dir: Path,
               years: list[str], out_dir: Path) -> None:
    cfg_base = OmegaConf.load(dataset_config_path)
    assert isinstance(cfg_base, DictConfig), "Dataset config must be a yaml mapping, not a list"
    out_dir.mkdir(parents=True, exist_ok=True)

    progress_path = out_dir / "progress.json"
    metadata_path = out_dir / "metadata.json"
    inputs_path   = out_dir / "inputs.npy"
    targets_path  = out_dir / "targets.npy"

    completed = json.loads(progress_path.read_text()) if progress_path.exists() else []
    years_todo = [y for y in years if y not in completed]
    if not years_todo:
        print("All years already completed.")
        return

    # Determine shapes and metadata from the first sample of the first todo year
    print(f"Probing shapes from year {years_todo[0]} ...")
    cfg_probe = deepcopy(cfg_base)
    cfg_probe.path = str(zarr_dir / f"zarr_{years_todo[0]}")
    cfg_probe.max_year = int(years_todo[0])
    probe_ds = ERA5Dataset(cfg_probe)
    sample_input, sample_target = probe_ds[0]
    input_shape  = sample_input.shape
    target_shape = sample_target.shape
    print(f"  Input shape: {input_shape}, Target shape: {target_shape}")

    # Save / verify metadata
    if metadata_path.exists():
        meta = json.loads(metadata_path.read_text())
        assert tuple(meta["input_shape"]) == input_shape,  "Input shape mismatch"
        assert tuple(meta["target_shape"]) == target_shape, "Target shape mismatch"
    else:
        meta = {
            "input_shape":  list(input_shape),
            "target_shape": list(target_shape),
            "pressure_levels": probe_ds.pressure_levels.tolist(),
            "grid_lon":  probe_ds.grid_lon.tolist(),
            "grid_lat":  probe_ds.grid_lat.tolist(),
            "means":      probe_ds.means.tolist(),
            "stds":       probe_ds.stds.tolist(),
            "diff_means": probe_ds.diff_means.tolist(),
            "diff_stds":  probe_ds.diff_stds.tolist(),
        }
    del probe_ds

    # Pre-count total N so we can size the memmaps upfront
    # (N will be written to metadata after all years are done)
    n_per_year: dict[str, int] = meta.get("n_per_year", {})

    for year in years_todo:
        print(f"\nYear {year} ...", flush=True)
        inputs_year, targets_year = samples_for_year(cfg_base, zarr_dir, year)
        n_per_year[year] = len(inputs_year)

        if inputs_path.exists():
            # Append to existing memmaps by resizing
            N_old = sum(n_per_year[y] for y in completed)
            N_new = N_old + len(inputs_year)
            inp_mm  = np.memmap(inputs_path,  dtype=np.float32, mode="r+",
                                shape=(N_old, *input_shape))
            tgt_mm  = np.memmap(targets_path, dtype=np.float32, mode="r+",
                                shape=(N_old, *target_shape))
            # Flush and reopen with extended shape
            del inp_mm, tgt_mm
            inp_mm  = np.memmap(inputs_path,  dtype=np.float32, mode="r+",
                                shape=(N_new, *input_shape))
            tgt_mm  = np.memmap(targets_path, dtype=np.float32, mode="r+",
                                shape=(N_new, *target_shape))
            inp_mm[N_old:] = inputs_year
            tgt_mm[N_old:] = targets_year
        else:
            inp_mm = np.memmap(inputs_path,  dtype=np.float32, mode="w+",
                               shape=(len(inputs_year), *input_shape))
            tgt_mm = np.memmap(targets_path, dtype=np.float32, mode="w+",
                               shape=(len(inputs_year), *target_shape))
            inp_mm[:] = inputs_year
            tgt_mm[:] = targets_year

        inp_mm.flush()
        tgt_mm.flush()
        del inp_mm, tgt_mm, inputs_year, targets_year

        completed.append(year)
        progress_path.write_text(json.dumps(completed))
        meta["n_per_year"] = n_per_year
        meta["N"] = sum(n_per_year.values())
        metadata_path.write_text(json.dumps(meta))
        print(f"  Year {year} written. Total samples so far: {meta['N']}", flush=True)

    print(f"\nDone. Total samples: {meta['N']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config", type=Path,
                        default=Path("configs/dataset/era5.yaml"))
    parser.add_argument("--zarr-dir", type=Path,
                        default=Path("./data/era5"))
    parser.add_argument("--years", nargs="+",
                        default=[str(y) for y in range(2015, 2025)])
    parser.add_argument("--out-dir", type=Path,
                        default=Path("./data/era5/precomputed"))
    args = parser.parse_args()
    precompute(args.dataset_config, args.zarr_dir, args.years, args.out_dir)
