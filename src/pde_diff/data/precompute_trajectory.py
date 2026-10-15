"""
One-time preprocessing of the held-out test years into an hourly state
trajectory for autoregressive rollout evaluation (see TrajectoryDataset).

Unlike precompute_dataset.py, nothing is stored per (t-1, t, t+1) triplet:
each hour is stored once, so any init time and horizon is a window into the
trajectory. Years are concatenated in time order and split into contiguous
segments wherever consecutive hours are more than 1 h apart, so a December
joins the following January-February (DJF winter) when both years are given.

Usage:
    python src/pde_diff/data/precompute_trajectory.py \
        --dataset-config configs/dataset/era5.yaml \
        --zarr-dir      ./data/era5 \
        --years         2023 2024 \
        --stats-from    ./data/era5/precomputed/metadata.json \
        --out-dir       ./data/era5/test_trajectory

Output layout:
    <out-dir>/
        states.npy      (T, C, lon, lat) float32, raw (unnormalised) states
        clock.npy       (T, 4, lon, lat) float32, clock features per hour
        times.npy       (T,) datetime64[h]
        segment_id.npy  (T,) int, contiguous-hour segment index
        metadata.json   grid, levels, features, segments and the training
                        normalisation stats (copied from --stats-from)
"""
import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf

from pde_diff.data.datasets import ERA5Dataset


def load_year(cfg_base: DictConfig, zarr_dir: Path, year: str) -> tuple[ERA5Dataset, np.ndarray, np.ndarray, np.ndarray]:
    """Load one year's zarr into RAM; return (dataset, raw states, clock, times)."""
    cfg = deepcopy(cfg_base)
    cfg.path = str(zarr_dir / f"zarr_{year}")
    cfg.max_year = int(year)
    dataset = ERA5Dataset(cfg)
    if dataset.static_features:
        raise NotImplementedError("static_features are not supported by the trajectory store")

    print(f"  {year}: loading into RAM ...", flush=True)
    dataset.data = dataset.data.load()
    states = dataset.raw_states(slice(None)).astype(np.float32)
    clock = dataset._generate_clock_features(dataset.data)
    times = dataset.data.time.values.astype("datetime64[h]")
    print(f"  {year}: {len(times)} hours, states {states.shape}", flush=True)
    return dataset, states, clock, times


def precompute(dataset_config_path: Path, zarr_dir: Path, years: list[str],
               out_dir: Path, stats_from: Path | None) -> None:
    cfg_base = OmegaConf.load(dataset_config_path)
    assert isinstance(cfg_base, DictConfig), "Dataset config must be a yaml mapping, not a list"
    out_dir.mkdir(parents=True, exist_ok=True)

    states, clock, times = [], [], []
    probe = None
    for year in sorted(years):
        dataset, s, c, t = load_year(cfg_base, zarr_dir, year)
        if probe is None:
            probe = dataset
        states.append(s)
        clock.append(c)
        times.append(t)
    states = np.concatenate(states)
    clock = np.concatenate(clock)
    times = np.concatenate(times)

    diffs = np.diff(times).astype(int)
    assert (diffs > 0).all(), "trajectory times must be strictly increasing"
    segment_id = np.concatenate([[0], np.cumsum(diffs > 1)])

    if stats_from is not None and stats_from.exists():
        train_meta = json.loads(stats_from.read_text())
        stats = {k: train_meta[k] for k in ("means", "stds", "diff_means", "diff_stds")}
        assert np.allclose(train_meta["grid_lon"], probe.grid_lon), "grid_lon differs from training data"
        assert np.allclose(train_meta["grid_lat"], probe.grid_lat), "grid_lat differs from training data"
        print(f"Normalisation stats copied from {stats_from}")
    else:
        print(f"WARNING: --stats-from {stats_from} not found; using the dataset's const.py stats")
        stats = {k: getattr(probe, k).tolist() for k in ("means", "stds", "diff_means", "diff_stds")}

    segments = []
    for seg in np.unique(segment_id):
        seg_times = times[segment_id == seg]
        segments.append({"id": int(seg), "start": str(seg_times[0]), "end": str(seg_times[-1]),
                         "n_hours": int(len(seg_times))})

    np.save(out_dir / "states.npy", states)
    np.save(out_dir / "clock.npy", clock)
    np.save(out_dir / "times.npy", times)
    np.save(out_dir / "segment_id.npy", segment_id)
    meta = {
        "years": sorted(years),
        "T": int(len(times)),
        "state_shape": list(states.shape[1:]),
        "atmospheric_features": list(probe.atmospheric_features),
        "single_features": list(probe.single_features),
        "pressure_levels": probe.pressure_levels.tolist(),
        "grid_lon": probe.grid_lon.tolist(),
        "grid_lat": probe.grid_lat.tolist(),
        "segments": segments,
        **stats,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=1))

    print(f"\nDone. {len(times)} hours in {len(segments)} segments -> {out_dir}")
    for seg in segments:
        print(f"  segment {seg['id']}: {seg['start']} -> {seg['end']} ({seg['n_hours']} h)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config", type=Path, default=Path("configs/dataset/era5.yaml"))
    parser.add_argument("--zarr-dir", type=Path, default=Path("./data/era5"))
    parser.add_argument("--years", nargs="+", default=["2023", "2024"])
    parser.add_argument("--stats-from", type=Path, default=Path("./data/era5/precomputed/metadata.json"),
                        help="Training metadata.json whose normalisation stats the test set must reuse")
    parser.add_argument("--out-dir", type=Path, default=Path("./data/era5/test_trajectory"))
    args = parser.parse_args()
    precompute(args.dataset_config, args.zarr_dir, args.years, args.out_dir, args.stats_from)
