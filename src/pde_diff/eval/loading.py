"""Shared model/dataset loading helpers for both single-model and
multi-model-comparison eval pipelines."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from pde_diff.utils import DatasetRegistry
from pde_diff.model import DiffusionModel
from pde_diff.eval_primitives import find_metrics_csv


def find_best_fold(model_path: str, fold_no: int = 5, log_path: str = "logs") -> tuple[str, float | None]:
    """Pick the fold with the lowest validation weighted-MSE for a k-fold
    cross-validated model, or fall back to a single (non-folded) model id."""
    base_id = model_path.split('/')[-1]

    if os.path.exists(os.path.join(log_path, base_id)):
        print("No folds found, using single model.")
        return (base_id, None)

    if os.path.exists(os.path.join(log_path, f"{base_id}-1")):
        best_model = (None, float("inf"))
        skipped = []
        for fold in range(1, fold_no + 1):
            current_model_id = f"{base_id}-{fold}"
            try:
                csv_path = find_metrics_csv(current_model_id, log_path=log_path)
            except FileNotFoundError as e:
                skipped.append(str(e))
                continue

            df = (
                pd.read_csv(csv_path)
                .apply(pd.to_numeric)
                .dropna(subset=["step"])
                .sort_values("step")
            )

            min_val = min(df["val_mse_(weighted)"].dropna().values)

            if min_val < best_model[1]:
                best_model = (current_model_id, min_val)

        if best_model[0] is None:
            raise FileNotFoundError(
                f"find_best_fold: found a '{base_id}-1' directory but no fold's metrics.csv "
                f"could be read for base id '{base_id}' under '{log_path}'. Details:\n"
                + "\n".join(skipped)
            )
        return best_model

    print("No logs found, using single model.")
    return (base_id, None)


def load_model_config(model_path: str, cfg: DictConfig | None = None) -> DictConfig:
    """Load a trained model's saved config.yaml, optionally merging eval-time
    overrides (`cfg`) on top of it."""
    model_cfg = OmegaConf.load(Path(model_path) / "config.yaml")
    if cfg is not None:
        model_cfg = OmegaConf.merge(model_cfg, cfg)
    return model_cfg


def load_model(cfg: DictConfig, model_path: str, device: str | None = None) -> DiffusionModel:
    """Instantiate a DiffusionModel from `cfg` and load its saved best-val
    weights from `model_path`."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg.model.dims = cfg.dataset.dims
    model = DiffusionModel(cfg)
    state_dict = torch.load(Path(model_path) / "best-val_loss-weights.pt", map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def build_test_dataloader(cfg: DictConfig, batch_size: int = 1, dataset_name: str | None = None) -> tuple[torch.utils.data.Dataset, DataLoader]:
    """Instantiate the (OOD/test) evaluation dataset and its DataLoader.

    `dataset_name`, if given, overrides `cfg.dataset.name` (e.g. to switch to
    the multi-step-ahead `"era5_test"` dataset for rollout evaluation).
    """
    dataset_cfg = cfg.dataset
    if dataset_name is not None:
        dataset_cfg = OmegaConf.merge(dataset_cfg, {"name": dataset_name})
    dataset = DatasetRegistry.create(dataset_cfg)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
    )
    return dataset, dataloader


def get_grid_lat(dataset) -> np.ndarray:
    """Read the dataset's real latitude coordinate array (used for PSD
    grid-spacing calibration), instead of hardcoding a resolution."""
    return np.asarray(dataset.grid_lat)
