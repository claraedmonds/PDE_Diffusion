"""Autoregressive OOD/test rollout forecasting evaluation: steps forward the
model `cfg.eval.steps` hours, computing per-step MSE + residual losses,
qualitative sample/residual plots, and MSE maps."""
import os
from pathlib import Path

import einops as ein
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from pde_diff.utils import LossRegistry
from pde_diff.model import DiffusionModel
from pde_diff.eval.residuals import build_residual_loss_fn
from pde_diff.eval.samples import (
    plot_forecasts_vs_targets,
    plot_residuals_with_truth,
    plot_recursive_residuals,
    plot_mse_of_vars,
    plot_mse_of_all_vars,
)
VAR_NAMES_ORDER = ["u", "v", "pv", "t", "z"]


def get_states(conds, state_changes):
    """Integrate a sequence of predicted state changes into absolute states,
    starting from the last known conditioning state."""
    states = []
    num_vars = (conds.shape[1] - 8) // 6
    next_state = conds[0, num_vars * 3 + 4:-4, :, :] + state_changes[0, :, :, :] if conds.shape[1] > 15 else conds[0, :, :, :] + state_changes[0, :, :, :]
    states.append(next_state.clone())
    for step in range(1, state_changes.shape[0]):
        next_state = next_state + state_changes[step, :, :, :]
        states.append(next_state.clone())
    return torch.stack(states, dim=0)


def evaluate_forecasting(
    model: DiffusionModel,
    dataset_test: Dataset,
    steps: int,
    dir: Path,
    n_qualitative_samples: int = 1,
    plot_sample_grids: bool = True,
) -> dict[str, dict[str, list[float]]]:
    """Autoregressively roll `model` forward `steps` hours on `dataset_test`,
    computing per-step MSE + residual losses (saved by the caller as CSVs),
    qualitative sample/residual plots for the first `n_qualitative_samples`
    samples, and full-dataset MSE maps."""
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=4, persistent_workers=True)

    vor_loss = build_residual_loss_fn(model, dataset_test)
    loss_fns = {
        "mse": LossRegistry.create(OmegaConf.create({"name": "mse"})),
        "residual": vor_loss,
    }

    loss_names = ["mse", "mse_change", "val_era5_sampled_planetary_residual(norm)", "val_era5_sampled_geo_wind_residual(norm)"]
    loss = {name: {str(i + 1): [] for i in range(steps)} for name in loss_names}

    mse_accum = None
    total_counts = 0

    for i, state in enumerate(tqdm(dataloader_test)):
        conditionals, targets, raw_target_states = state
        conditionals = conditionals.to(model.device)
        targets = targets[0].to(model.device)
        raw_target_states = raw_target_states[0].to(model.device)

        forecasted_changes = model.forecast(conditionals, steps=steps)[0]

        conditionals_rearranged = ein.rearrange(conditionals, "b (state var) lon lat -> b state var lon lat", state=2)
        un_norm_cond = dataset_test._unnormalize(conditionals_rearranged[:, 1, :15].detach().cpu(), dataset_test.means, dataset_test.stds)
        un_norm_forecasted_changes = dataset_test._unnormalize(forecasted_changes.detach().cpu(), dataset_test.diff_means, dataset_test.diff_stds)
        un_norm_forecasted_states = get_states(un_norm_cond, un_norm_forecasted_changes)
        norm_forecasted_state = dataset_test._normalize(un_norm_forecasted_states.detach().cpu(), dataset_test.means, dataset_test.stds)
        prev_states = torch.cat([conditionals[:, 19:-4], norm_forecasted_state[:-1].to(model.device)], dim=0)

        un_norm_targets = dataset_test._unnormalize(targets.detach().cpu(), dataset_test.diff_means, dataset_test.diff_stds)
        un_norm_target_states = get_states(un_norm_cond, un_norm_targets)
        norm_target_states = dataset_test._normalize(un_norm_target_states.detach().cpu(), dataset_test.means, dataset_test.stds)
        prev_states_true = torch.cat([conditionals[:, 19:-4], norm_target_states[:-1].to(model.device)], dim=0)

        un_norm_forecasted_states = ein.rearrange(un_norm_forecasted_states, "b (var lev) lon lat -> b lev var lon lat", lev=3)
        un_norm_target_states = ein.rearrange(un_norm_target_states, "b (var lev) lon lat -> b lev var lon lat", lev=3)
        un_norm_targets = ein.rearrange(un_norm_targets, "b (var lev) lon lat -> b lev var lon lat", lev=3)
        un_norm_forecasted_changes = ein.rearrange(un_norm_forecasted_changes, "b (var lev) lon lat -> b lev var lon lat", lev=3)

        raw_target_states = ein.rearrange(raw_target_states, "b (var lev) lon lat -> b lev var lon lat", lev=3)
        assert torch.allclose(raw_target_states.detach().cpu(), un_norm_target_states, atol=1e-5, rtol=1e-6), "Problem, norm back and forth changing values too much"

        for loss_name, loss_fn in loss_fns.items():
            if loss_name == "mse":
                for k in range(steps):
                    l = loss_fn(norm_forecasted_state[k], norm_target_states[k])
                    loss["mse"][str(k + 1)].append(l.item())

                    l = loss_fn(forecasted_changes[k], targets[k])
                    loss["mse_change"][str(k + 1)].append(l.item())
            else:
                loss_geo_wind = loss_fn.compute_residual_geostrophic_wind(x0_previous=prev_states, x0_change_pred=forecasted_changes, normalize=True).abs()
                mean_loss_geo_wind = loss_geo_wind.mean(dim=(1, 2, 3))
                loss_planetary = loss_fn.compute_residual_planetary_vorticity(x0_previous=prev_states, x0_change_pred=forecasted_changes, normalize=True).abs()
                mean_loss_planetary = loss_planetary.mean(dim=(1, 2))
                for k in range(steps):
                    loss["val_era5_sampled_planetary_residual(norm)"][str(k + 1)].append(mean_loss_planetary[k].item())
                    loss["val_era5_sampled_geo_wind_residual(norm)"][str(k + 1)].append(mean_loss_geo_wind[k].item())

        targets_rearranged = ein.rearrange(targets, "b (lev var) lon lat -> b lev var lon lat", lev=3)
        forecasted_changes_rearranged = ein.rearrange(forecasted_changes, "b (lev var) lon lat -> b lev var lon lat", lev=3)
        sq_err = (targets_rearranged[0, 1] - forecasted_changes_rearranged[0, 1]) ** 2

        if mse_accum is None:
            mse_accum = sq_err.detach().cpu().numpy()
        else:
            mse_accum += sq_err.detach().cpu().numpy()
        total_counts += 1

        if plot_sample_grids and i < n_qualitative_samples:
            dir_sample = os.path.join(dir, f"sample_{i}")
            os.makedirs(dir_sample, exist_ok=True)
            lvl = 1

            loss_geo_wind_target = loss_fn.compute_residual_geostrophic_wind(x0_previous=prev_states_true, x0_change_pred=targets, normalize=True).abs()
            loss_planetary_target = loss_fn.compute_residual_planetary_vorticity(x0_previous=prev_states_true, x0_change_pred=targets, normalize=True).abs()

            # Color limits are computed dynamically per-plot (limits=None) rather
            # than pinned to magic numbers from one past run.
            plot_residuals_with_truth(loss_geo_wind[0, lvl], loss_geo_wind_target[0, lvl], "gw", sample_idx=i, dir=dir_sample)
            plot_residuals_with_truth(loss_planetary[0], loss_planetary_target[0], "plan", sample_idx=i, dir=dir_sample)
            if steps > 1:
                plot_recursive_residuals(loss_planetary, loss_planetary_target, "plan", sample_idx=i, dir=dir_sample)
                plot_recursive_residuals(loss_geo_wind, loss_geo_wind_target, "gw", sample_idx=i, dir=dir_sample)

            for j, var in enumerate(VAR_NAMES_ORDER):
                plot_forecasts_vs_targets(
                    forecasts=[un_norm_forecasted_states.detach().cpu()[k, lvl, j, :, :] for k in range(un_norm_forecasted_states.shape[0])],
                    targets=[raw_target_states.detach().cpu()[k, lvl, j, :, :] for k in range(un_norm_target_states.shape[0])],
                    variable=var,
                    sample_idx=i,
                    dir=dir_sample,
                )
                plot_forecasts_vs_targets(
                    forecasts=[un_norm_forecasted_changes.detach().cpu()[k, lvl, j, :, :] for k in range(un_norm_forecasted_states.shape[0])],
                    targets=[un_norm_targets.detach().cpu()[k, lvl, j, :, :] for k in range(un_norm_target_states.shape[0])],
                    variable=var,
                    sample_idx=i,
                    dir=dir_sample,
                    states=False,
                )

    if mse_accum is not None and total_counts > 0:
        mse_maps = mse_accum / float(total_counts)
        mse_out_dir = Path(dir) / "mse_maps"
        mse_out_dir.mkdir(parents=True, exist_ok=True)
        for j, var in enumerate(VAR_NAMES_ORDER):
            plot_mse_of_vars(mse_maps[j], out_dir=mse_out_dir, var=var)
        plot_mse_of_all_vars(mse_maps, out_dir=mse_out_dir, vars=VAR_NAMES_ORDER)

    return loss
