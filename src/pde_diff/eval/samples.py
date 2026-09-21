"""Sample-level evaluation plots: prediction-vs-target-vs-diff grids,
recursive forecast grids, residual maps, MSE maps, and error-distribution
histograms."""
import os
from pathlib import Path

import einops as ein
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torch.utils.data import DataLoader
from tqdm import tqdm

from pde_diff.eval_primitives import EXTENT_SUBSET, COLOR_BARS, VAR_UNITS, PLOT_TYPE

# Eval-specific short symbols for LaTeX titles/axis labels. Distinct from
# pde_diff.eval_primitives.VAR_NAMES (which uses full LaTeX macros like
# r"$q_E$") to avoid the two coexisting under the same name.
EVAL_VAR_SYMBOLS = {
    "u": "u",
    "v": "v",
    "t": "T",
    "z": "\\Phi",
    "pv": "q_{E,}",
}

VAR_FULL_NAMES = {
    "u": "Eastward Wind",
    "v": "Northward Wind",
    "t": "Temperature",
    "z": "Geopotential",
    "pv": "Ertel Pot. Vort.",
}

RES_NAMES = {
    "plan": ("Planetary Vorticity", 1, r"$s^{-2}$"),
    "gw": ("Geostrophic Wind", 2, r"$m \cdot s^{-1}$"),
}


def plot_sample_target_absdiff_stacked(
    sample: torch.Tensor,
    target: torch.Tensor,
    variable: str = "t",
    sample_idx: int = 0,
    dir=Path("./reports/figures/samples"),
):
    sample = sample.detach().cpu().numpy()
    target = target.detach().cpu().numpy()

    vmin = min(sample.min(), target.min())
    vmax = max(sample.max(), target.max())

    fig = plt.figure(figsize=(8, 2.4), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.00, h_pad=0.00, wspace=0.01, hspace=0.00)

    gs = fig.add_gridspec(nrows=3, ncols=2, width_ratios=[40, 1], height_ratios=[1, 1, 1])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])

    cax_top = fig.add_subplot(gs[0:2, 1])
    cax_bot = fig.add_subplot(gs[2, 1])
    var = EVAL_VAR_SYMBOLS.get(variable, variable)
    var_name = VAR_FULL_NAMES.get(variable, variable)

    im1 = ax1.imshow(sample.T, cmap=COLOR_BARS.get(variable, "coolwarm"), extent=EXTENT_SUBSET, vmin=vmin, vmax=vmax, aspect="auto")
    ax1.set_title(rf"Prediction of {var_name}, $\tilde{{{var}}}_0$")
    ax1.set_xticklabels([])
    ax1.set_xlim(EXTENT_SUBSET[0], EXTENT_SUBSET[1])
    ax1.set_ylim(EXTENT_SUBSET[2], EXTENT_SUBSET[3])

    ax2.imshow(target.T, cmap=COLOR_BARS.get(variable, "coolwarm"), extent=EXTENT_SUBSET, vmin=vmin, vmax=vmax, aspect="auto")
    ax2.set_title(rf"True state of {var_name}, ${{{var}}}_0$")
    ax2.set_xticklabels([])
    ax2.set_xlim(EXTENT_SUBSET[0], EXTENT_SUBSET[1])
    ax2.set_ylim(EXTENT_SUBSET[2], EXTENT_SUBSET[3])

    plt.colorbar(im1, cax=cax_top, label=f"{var}")

    diff = np.abs(target - sample)
    im3 = ax3.imshow(diff.T, cmap="Oranges", extent=EXTENT_SUBSET, aspect="auto")
    ax3.set_title(rf"MAE of {var_name}, $|{{{var}}}_0-\tilde{{{var}}}_0|$")
    ax3.set_xlim(EXTENT_SUBSET[0], EXTENT_SUBSET[1])
    ax3.set_ylim(EXTENT_SUBSET[2], EXTENT_SUBSET[3])

    plt.colorbar(im3, cax=cax_bot, label=f"{VAR_UNITS.get(variable, variable)}")

    for ax in (ax1, ax2, ax3):
        ax.set_yticks([])

    os.makedirs(dir, exist_ok=True)
    out_path = os.path.join(dir, f"sample_w_target_diff_{sample_idx}_{variable}{PLOT_TYPE}")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved plot to {out_path}")


def plot_forecasts_vs_targets(
    forecasts: list,
    targets: list,
    variable: str = "t",
    sample_idx: int = 0,
    dir=Path("./reports/figures/samples"),
    states=True,
    limits=None,
):
    """Plot recursive forecasts vs targets. Left column: forecasts. Right
    column: targets. If `limits` is None, color scales are computed
    dynamically from the data instead of relying on caller-supplied bounds."""
    assert len(forecasts) == len(targets), "forecasts and targets must have the same length"

    forecasts = [f.detach().cpu().numpy() if hasattr(f, "detach") else f for f in forecasts]
    targets = [t.detach().cpu().numpy() if hasattr(t, "detach") else t for t in targets]
    differences = [np.abs(t - f) for f, t in zip(forecasts, targets)]

    vmin = min(f.min() for f in forecasts + targets) if not limits else limits[0]
    vmax = max(f.max() for f in forecasts + targets) if not limits else limits[1]
    vmin_d = min(f.min() for f in differences) if not limits else limits[2]
    vmax_d = max(f.max() for f in differences) if not limits else limits[3]

    n_steps = len(forecasts)
    fig = plt.figure(figsize=(10, 2.4 * n_steps / 5), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.01, wspace=0.02, hspace=0.02)

    gs = fig.add_gridspec(nrows=n_steps, ncols=5, width_ratios=[1, 1, 0.05, 1, 0.05], hspace=0.00, wspace=0.05)

    cmap = COLOR_BARS.get(variable, "coolwarm")

    axes_forecast, axes_target, axes_diff = [], [], []

    var = EVAL_VAR_SYMBOLS.get(variable, variable)
    var_name = VAR_FULL_NAMES.get(variable, variable)

    for i in range(n_steps):
        ax_f = fig.add_subplot(gs[i, 0])
        ax_t = fig.add_subplot(gs[i, 1])
        ax_d = fig.add_subplot(gs[i, 3])

        im = ax_f.imshow(forecasts[i].T, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", extent=EXTENT_SUBSET)
        ax_t.imshow(targets[i].T, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", extent=EXTENT_SUBSET)
        im_diff = ax_d.imshow(differences[i].T, vmin=vmin_d, vmax=vmax_d, cmap="Oranges", aspect="auto", extent=EXTENT_SUBSET)

        for axs in (ax_f, ax_t, ax_d):
            axs.set_xlim(EXTENT_SUBSET[0], EXTENT_SUBSET[1])
            axs.set_ylim(EXTENT_SUBSET[2], EXTENT_SUBSET[3])
            axs.set_yticks([])
            if i < n_steps - 1:
                axs.set_xticklabels([])
        ax_f.set_ylabel(f"{i+1} h", rotation=0, labelpad=15, va="center")

        axes_forecast.append(ax_f)
        axes_target.append(ax_t)
        axes_diff.append(ax_d)

    if states:
        axes_forecast[0].set_title(rf"Forecasts of {var_name}, $\tilde{{{var}}}_0$")
        axes_target[0].set_title(rf"True states of {var_name}, ${{{var}}}_0$")
        axes_diff[0].set_title(rf"MAE, $|{{{var}}}_0-\tilde{{{var}}}_0|$")
    else:
        axes_forecast[0].set_title(rf"Forecasts of {var_name}, $\Delta \tilde{{{var}}}_0$")
        axes_target[0].set_title(rf"True changes of {var_name}, $\Delta {{{var}}}_0$")
        axes_diff[0].set_title(rf"MAE, $|\Delta {{{var}}}_0- \Delta \tilde{{{var}}}_0|$")

    cax = fig.add_subplot(gs[:, 2])
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(f"{VAR_UNITS.get(variable, variable)}", labelpad=0)
    if variable in ("z", "pv"):
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))
        cbar.formatter = formatter
        cbar.update_ticks()

    cax2 = fig.add_subplot(gs[:, 4])
    cbar2 = plt.colorbar(im_diff, cax=cax2)
    if variable == "pv":
        formatter2 = ticker.ScalarFormatter(useMathText=True)
        formatter2.set_scientific(True)
        formatter2.set_powerlimits((0, 0))
        cbar2.formatter = formatter2
        cbar2.update_ticks()

    suffix = "" if states else "changes"
    os.makedirs(dir, exist_ok=True)
    out_path = os.path.join(dir, f"forecasts_vs_targets_w_diff_sample_{sample_idx}_{variable}{suffix}{PLOT_TYPE}")

    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    print(f"Saved forecast vs target plot to {out_path}")


def plot_forecast_error_distributions(
    forecasts,
    targets,
    variable: str = "t",
    dir=Path("./reports/figures/"),
    bins: int = 120,
    percentiles=(1.0, 99.0),
):
    """Histograms of (forecast-change - target-change) per pressure level."""
    assert len(forecasts) == len(targets), "forecasts and targets must have same length"

    var = EVAL_VAR_SYMBOLS.get(variable, variable)

    errors_per_lvl = [[] for _ in range(3)]
    for lvl in range(3):
        for f, t in zip(forecasts[lvl], targets[lvl]):
            if hasattr(f, "detach"):
                f = f.detach().cpu().numpy()
            if hasattr(t, "detach"):
                t = t.detach().cpu().numpy()
            err = (np.asarray(f) - np.asarray(t)).flatten()
            err = err[np.isfinite(err)]
            errors_per_lvl[lvl].append(err)

    all_errors = np.concatenate([e for e in errors_per_lvl])

    low_p, high_p = percentiles
    if all_errors.size > 2:
        lo, hi = np.percentile(all_errors, [low_p, high_p])
    else:
        lo, hi = float(all_errors.min()), float(all_errors.max())
    if lo == hi:
        lo -= 1e-6
        hi += 1e-6

    fig, ax = plt.subplots(figsize=(3.8, 3.8))
    colors = ["#2A9D8F", "#E76F51", "#1A4DAC"]
    levels = ["450hPa", "500hPa", "550hPa"]
    for lvl in range(3):
        data = np.array(errors_per_lvl[lvl]).flatten()

        mean = float(data.mean())
        std = float(data.std())

        ax.hist(data, bins=bins, range=(lo, hi), histtype="stepfilled", alpha=0.6, edgecolor="black", color=colors[lvl], label=f"{levels[lvl]}: Mean={mean:.2e},Std={std:.2e}")
        ax.axvline(mean, linestyle="-", linewidth=2, color=colors[lvl])
        ax.axvline(mean - std, linestyle="--", linewidth=1.2, color=colors[lvl])
        ax.axvline(mean + std, linestyle="--", linewidth=1.2, color=colors[lvl])

    ax.set_title(rf"Prediction errors of ${var}$" if variable != "pv" else r"Forecast errors of $q_{E}$")
    ax.set_xlabel(rf"$\Delta \tilde{{{var}}}_0 - \Delta {{{var}}}_0$")
    ax.set_xlim(lo, hi)
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    os.makedirs(dir, exist_ok=True)
    out_path = os.path.join(dir, f"forecast_error_distribution_{variable}{PLOT_TYPE}")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved forecast error distribution plot to {out_path}")


def plot_residuals_with_truth(
    residual_pred: torch.Tensor,
    residual_true: torch.Tensor,
    name: str,
    sample_idx: int = 0,
    show_difference: bool = True,
    dir=Path("./reports/figures/samples"),
    limits=None,
):
    """Plot residual maps from forecast and true state (dynamic vmin/vmax
    computed from the data when `limits` is None)."""
    res_name, res_number, res_unit = RES_NAMES[name]

    residual_pred = residual_pred.detach().cpu().numpy()
    residual_true = residual_true.detach().cpu().numpy()

    vmin = min(residual_pred.min(), residual_true.min()) if not limits else limits[0]
    vmax = max(residual_pred.max(), residual_true.max()) if not limits else limits[1]

    nrows = 3 if show_difference else 2

    fig = plt.figure(figsize=(5, 2.4 if show_difference else 1.6), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.00, h_pad=0.00, wspace=0.01, hspace=0.00)

    gs = fig.add_gridspec(nrows=nrows, ncols=2, width_ratios=[40, 1], height_ratios=[1] * nrows)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])

    if show_difference:
        ax3 = fig.add_subplot(gs[2, 0])
        cax_top = fig.add_subplot(gs[0:2, 1])
        cax_bot = fig.add_subplot(gs[2, 1])
    else:
        cax_top = fig.add_subplot(gs[:, 1])

    im1 = ax1.imshow(residual_pred.T, cmap="magma", extent=EXTENT_SUBSET, vmin=vmin, vmax=vmax, aspect="auto")
    ax1.set_title(rf"{res_name} residual on forecast $\mathcal{{R}}_{{\mathrm{{{res_number}}}}}(\tilde{{\mathbf{{x}}}}_0)$")

    ax2.imshow(residual_true.T, cmap="magma", extent=EXTENT_SUBSET, vmin=vmin, vmax=vmax, aspect="auto")
    ax2.set_title(rf"{res_name} residual on true state $\mathcal{{R}}_{{\mathrm{{{res_number}}}}}(\mathbf{{x}}_0)$")

    plt.colorbar(im1, cax=cax_top, label=res_unit)

    if show_difference:
        diff = np.abs(residual_pred - residual_true)
        vmin_diff = diff.min() if not limits else limits[2]
        vmax_diff = diff.max() if not limits else limits[3]
        im3 = ax3.imshow(diff.T, cmap="Oranges", extent=EXTENT_SUBSET, aspect="auto", vmin=vmin_diff, vmax=vmax_diff)
        ax3.set_title(rf"Residual difference $|\mathcal{{R}}_{{\mathrm{{{res_number}}}}}(\tilde{{\mathbf{{x}}}}_0) - \mathcal{{R}}_{{\mathrm{{{res_number}}}}}(\mathbf{{x}}_0)|$")
        plt.colorbar(im3, cax=cax_bot, label=res_unit)

    for ax in (ax1, ax2) if not show_difference else (ax1, ax2, ax3):
        ax.set_yticks([])
        ax.set_xlim(EXTENT_SUBSET[0], EXTENT_SUBSET[1])
        ax.set_ylim(EXTENT_SUBSET[2], EXTENT_SUBSET[3])

    for ax in (ax1,) if not show_difference else (ax1, ax2):
        ax.set_xticklabels([])

    suffix = "with_diff" if show_difference else "no_diff"
    os.makedirs(dir, exist_ok=True)
    out_path = os.path.join(dir, f"residual_{res_name.replace(' ', '_')}_{suffix}_{sample_idx}{PLOT_TYPE}")

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved residual plot to {out_path}")


def plot_recursive_residuals(
    residuals_pred: list,
    residuals_true: list,
    name: str,
    sample_idx: int = 0,
    dir=Path("./reports/figures/samples"),
    limits=None,
):
    """Plot recursive residuals on forecast vs true state, across steps."""
    assert len(residuals_pred) == len(residuals_true), "residuals_pred and residuals_true must have the same length"

    res_name, res_number, res_unit = RES_NAMES[name]

    residuals_pred = [r.detach().cpu().numpy() if hasattr(r, "detach") else r for r in residuals_pred]
    residuals_true = [r.detach().cpu().numpy() if hasattr(r, "detach") else r for r in residuals_true]

    diffs = [np.abs(p - t) for p, t in zip(residuals_pred, residuals_true)]

    vmin = min(r.min() for r in residuals_pred + residuals_true) if not limits else limits[0]
    vmax = max(r.max() for r in residuals_pred + residuals_true) if not limits else limits[1]
    vmin_d = min(d.min() for d in diffs) if not limits else limits[2]
    vmax_d = max(d.max() for d in diffs) if not limits else limits[3]

    n_steps = len(residuals_pred)
    fig = plt.figure(figsize=(10, 2.4 * n_steps / 5), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.01, wspace=0.02, hspace=0.02)

    gs = fig.add_gridspec(nrows=n_steps, ncols=5, width_ratios=[1, 1, 0.05, 1, 0.05], hspace=0.00, wspace=0.05)

    axes_pred, axes_true, axes_diff = [], [], []

    for i in range(n_steps):
        ax_p = fig.add_subplot(gs[i, 0])
        ax_t = fig.add_subplot(gs[i, 1])
        ax_d = fig.add_subplot(gs[i, 3])

        im = ax_p.imshow(residuals_pred[i].T, cmap="magma", vmin=vmin, vmax=vmax, aspect="auto", extent=EXTENT_SUBSET)
        ax_t.imshow(residuals_true[i].T, cmap="magma", vmin=vmin, vmax=vmax, aspect="auto", extent=EXTENT_SUBSET)
        im_diff = ax_d.imshow(diffs[i].T, cmap="Oranges", vmin=vmin_d, vmax=vmax_d, aspect="auto", extent=EXTENT_SUBSET)

        for axs in (ax_p, ax_t, ax_d):
            axs.set_xlim(EXTENT_SUBSET[0], EXTENT_SUBSET[1])
            axs.set_ylim(EXTENT_SUBSET[2], EXTENT_SUBSET[3])
            axs.set_yticks([])
            if i < n_steps - 1:
                axs.set_xticklabels([])

        ax_p.set_ylabel(f"Step {i+1}", rotation=0, labelpad=15, va="center")

        axes_pred.append(ax_p)
        axes_true.append(ax_t)
        axes_diff.append(ax_d)

    axes_pred[0].set_title(rf"{res_name} residual on forecast $\mathcal{{R}}_{{\mathrm{{{res_number}}}}}(\tilde{{\mathbf{{x}}}}_0)$")
    axes_true[0].set_title(rf"{res_name} residual on true state $\mathcal{{R}}_{{\mathrm{{{res_number}}}}}(\mathbf{{x}}_0)$")
    axes_diff[0].set_title(r"Residual MAE $|\mathcal{R}(\tilde{\mathbf{x}}_0) - \mathcal{R}(\mathbf{x}_0)|$")

    cax = fig.add_subplot(gs[:, 2])
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(res_unit, labelpad=0)

    cax2 = fig.add_subplot(gs[:, 4])
    cbar2 = plt.colorbar(im_diff, cax=cax2)
    cbar2.set_label(res_unit, labelpad=0)

    os.makedirs(dir, exist_ok=True)
    out_path = os.path.join(dir, f"recursive_residual_{res_name.replace(' ', '_')}_sample_{sample_idx}{PLOT_TYPE}")

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved recursive residual plot to {out_path}")


def plot_mse_of_vars(mse_map, out_dir, var):
    plt.figure(figsize=(6, 3))
    im = plt.imshow(mse_map.T, extent=EXTENT_SUBSET, origin="lower", cmap="viridis")
    plt.title(f"MSE map — {var} (500 hPa)")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"mse_map_{var}{PLOT_TYPE}"
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved MSE map for {var} to {out_path}")


def plot_mse_of_all_vars(mse_maps, out_dir, vars):
    fig = plt.figure(figsize=(4.3, 2.7))

    gs = fig.add_gridspec(nrows=len(vars), ncols=2, width_ratios=[40, 1], height_ratios=[1] * len(vars))
    for i, var in enumerate(vars):
        var_symbol = EVAL_VAR_SYMBOLS.get(var, var)
        ax = fig.add_subplot(gs[i, 0])

        im = ax.imshow(mse_maps[i].T, extent=EXTENT_SUBSET, origin="lower", cmap="viridis")
        ax.set_title(rf"Test MSE of $\small{{{{{var_symbol}}} , ||\Delta \tilde{{{var_symbol}}}_0 - \Delta {{{var_symbol}}}_0||^2}}$")
        ax.set_yticklabels([])
        ax.set_yticks([])
        if i < len(vars) - 1:
            ax.set_xticklabels([])
            ax.set_xticks([])
        ax.set_xlim(EXTENT_SUBSET[0], EXTENT_SUBSET[1])
        ax.set_ylim(EXTENT_SUBSET[3], EXTENT_SUBSET[2])

    cax = fig.add_subplot(gs[:, 1])
    plt.colorbar(im, cax=cax, label="MSE")
    fig.subplots_adjust(left=0.05, right=0.95, top=0.94, bottom=0.06, hspace=0.00, wspace=0.02)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"mse_map_all_variables{PLOT_TYPE}"
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved MSE map for all variables to {out_path}")


def generate_prediction_error_distributions(
    model,
    dataset_test,
    var_names: list[str] | None = None,
    max_hist_samples_per_step: int = 8,
    out_dir=Path("reports/eval/forecast_error_distributions"),
):
    """Collect single-step forecast-change vs target-change error samples
    across the test dataset and save one PNG per variable."""
    if var_names is None:
        var_names = ["u", "v", "pv", "t", "z"]

    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=4, persistent_workers=True)

    collectors_f = {v: [[] for _ in range(3)] for v in var_names}
    collectors_t = {v: [[] for _ in range(3)] for v in var_names}

    for state in tqdm(dataloader_test, desc="Collecting forecast error samples"):
        conditionals, targets, _ = state
        conditionals = conditionals.to(model.device)
        targets = targets[0].to(model.device)

        pred_changes = model.sample_loop(batch_size=1, conditionals=conditionals[None, 0])

        targets_rearranged = ein.rearrange(targets, "b (lev var) lon lat -> b lev var lon lat", lev=3)
        pred_changes_rearranged = ein.rearrange(pred_changes, "b (lev var) lon lat -> b lev var lon lat", lev=3)

        for j, var in enumerate(var_names):
            if all(len(collectors_f[var][lvl]) >= max_hist_samples_per_step for lvl in range(3)):
                continue

            f_arr = pred_changes_rearranged[0, :, j, :, :].detach().cpu()
            t_arr = targets_rearranged[0, :, j, :, :].detach().cpu()

            for lvl in range(3):
                mask = np.isfinite(f_arr[lvl].numpy().ravel()) & np.isfinite(t_arr[lvl].numpy().ravel())
                if mask.sum() == 0:
                    continue
                collectors_f[var][lvl].append(f_arr[lvl].numpy().ravel()[mask])
                collectors_t[var][lvl].append(t_arr[lvl].numpy().ravel()[mask])

        all_full = all(len(collectors_f[v][lvl]) >= max_hist_samples_per_step for v in var_names for lvl in range(3))
        if all_full:
            break

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for var in var_names:
        if len(collectors_f[var]) == 0:
            preds = [np.array([])]
            targs = [np.array([])]
        else:
            preds = [np.concatenate(collectors_f[var][lvl]) for lvl in range(3)]
            targs = [np.concatenate(collectors_t[var][lvl]) for lvl in range(3)]

        try:
            plot_forecast_error_distributions(preds, targs, variable=var, dir=out_dir)
        except Exception as e:
            print(f"Failed to plot forecast error distributions for var {var}: {e}")
