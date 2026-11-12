"""
Radial power-spectral-density (PSD) of rollout forecasts vs. ERA5 truth at
one pressure level, averaged over a fixed, evenly spaced subset of rollout
inits and every ensemble member, at a few lead times.

`compute_rollout_psd` saves the mean spectra per model to `psd_rollout.npz`
in its eval dir; `plot_psd_grid` draws one or more models' saved spectra on
one figure (rows = leads, columns = variables), so compare mode can overlay
models without re-running them.
"""
from pathlib import Path

import numpy as np
import torch
from scipy.fft import fft2, fftfreq
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from pde_diff.eval.palette import get_model_color
import einops as ein

from pde_diff.eval.residuals import build_residual_evaluator
from pde_diff.eval.rollout import Normalizer, rollout_ensemble, QUALITATIVE_LEVEL_IDX

EARTH_RADIUS_KM = 6371.0088
DEG_TO_KM = np.pi / 180.0 * EARTH_RADIUS_KM  # ~111.32 km/deg
PSD_FILE = "psd_rollout.npz"
PSD_KINDS = {"change": "1 h change", "state": "state",
             "geo": "geostrophic wind V_g", "ageo": "ageostrophic wind V - V_g"}
WIND_KINDS = ("geo", "ageo")  # columns: u, v components (see wind_decomposition)


def wind_decomposition(residual_fn, states: torch.Tensor, n_levels: int, level_idx: int, var_names: list[str]):
    """Geostrophic (u_g, v_g) and ageostrophic (u - u_g, v - v_g) wind of raw
    states (..., C, lon, lat) at `level_idx`, with the same QG V_g as R2.
    Returns two tensors (..., 2, lon, lat)."""
    lead_shape = states.shape[:-3]
    x = ein.rearrange(states.flatten(0, -4), "n (var lev) lon lat -> n lev var lon lat", lev=n_levels)
    u, v, z = (x[:, :, var_names.index(name)] for name in ("u", "v", "z"))
    u_g, v_g = residual_fn.geostrophic_wind(z)
    geo = torch.stack([u_g, v_g], dim=1)[:, :, level_idx]
    ageo = torch.stack([u - u_g, v - v_g], dim=1)[:, :, level_idx]
    return geo.view(*lead_shape, *geo.shape[1:]), ageo.view(*lead_shape, *ageo.shape[1:])


def radial_psd(fields: np.ndarray, dx_km: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Azimuthally-averaged (radial) power spectral density of one or a batch
    of 2-D fields, shaped (..., ny, nx).

    A Hann window is applied before the FFT. Without it, the field's sharp
    rectangular edges act as an implicit boxcar window, whose FFT leaks
    broadband power into every wavenumber bin (spectral leakage) — this masks
    real high-wavenumber differences between fields (e.g. it can make a
    heavily-blurred field's PSD look nearly identical to the unblurred one).

    Returns
    -------
    k_centres : np.ndarray — wavenumber bin centres [cycles / km]
    psd       : np.ndarray — mean power in each bin, shaped (..., n_bins)
    """
    fields = np.asarray(fields, dtype=np.float64)
    if np.isnan(fields).any():
        fill = np.nanmean(fields, axis=(-2, -1), keepdims=True)
        fields = np.where(np.isnan(fields), fill, fields)

    ny, nx = fields.shape[-2:]

    window = np.hanning(ny)[:, None] * np.hanning(nx)[None, :]
    window /= np.sqrt(np.mean(window ** 2))  # preserve overall power scale

    f2d = fft2((fields - fields.mean(axis=(-2, -1), keepdims=True)) * window, axes=(-2, -1))
    power = (np.abs(f2d) ** 2) / (ny * nx)

    KX, KY = np.meshgrid(fftfreq(nx, d=dx_km), fftfreq(ny, d=dx_km))
    K = np.sqrt(KX ** 2 + KY ** 2)

    n_bins = min(ny, nx) // 2
    k_bins = np.linspace(0, 0.5 / dx_km, n_bins + 1)
    bin_idx = np.digitize(K, k_bins).ravel() - 1  # bin i holds k_bins[i] <= K < k_bins[i + 1]
    in_range = bin_idx < n_bins
    onehot = np.zeros((ny * nx, n_bins))
    onehot[np.flatnonzero(in_range), bin_idx[in_range]] = 1.0
    counts = onehot.sum(0)
    valid = counts > 0

    psd = power.reshape(*power.shape[:-2], ny * nx) @ (onehot[:, valid] / counts[valid])
    k_centres = 0.5 * (k_bins[:-1] + k_bins[1:])
    return k_centres[valid], psd


def compute_dx_km(grid_lat: np.ndarray) -> float:
    """
    Isotropic grid spacing [km] derived from the actual latitude spacing of
    the dataset. The ERA5 crop is a regular lat-lon grid, not equal-area, so
    longitude spacing shrinks with cos(latitude); using the (constant)
    latitude-direction spacing for both FFT axes is an approximation. This
    only affects the absolute km/wavelength calibration on the x-axis — since
    the same dx_km is applied identically to ground truth and every model
    being compared, it does not distort the relative comparison between them.
    """
    grid_spacing_deg = float(np.abs(np.diff(np.sort(np.unique(grid_lat)))).mean())
    return grid_spacing_deg * DEG_TO_KM


@torch.no_grad()
def compute_rollout_psd(
    model,
    dataset,
    out_path: Path,
    n_inits: int = 50,
    n_members: int = 9,
    leads: list[int] = (1, 6, 12),
    level_idx: int = QUALITATIVE_LEVEL_IDX,
    inits_per_batch: int = 1,
    num_workers: int = 4,
) -> dict:
    """Mean radial PSD of the raw predicted state and 1 h state change at
    `level_idx`, over `n_inits` evenly spaced inits of `dataset` (a
    TrajectoryDataset) x `n_members` members, at each lead in `leads`; the
    ERA5 truth is averaged over the same inits. The init subset only depends
    on `len(dataset)`, so every model is scored on the same inits.

    Saved to `out_path`; an existing file with the same settings is reused."""
    leads = sorted(int(k) for k in leads if 1 <= k <= dataset.horizon)
    settings = dict(n_inits=n_inits, n_members=n_members, leads=leads, level_idx=level_idx)
    out_path = Path(out_path)
    if out_path.exists():
        saved = load_psd(out_path)
        has_all_kinds = all(f"{who}_{kind}" in saved for who in ("pred", "true") for kind in PSD_KINDS)
        if has_all_kinds and all(np.array_equal(saved.get(key), val) for key, val in settings.items()):
            print(f"Reusing PSD in {out_path}")
            return saved

    device = model.device
    norm = Normalizer(dataset, device)
    residual_fn = build_residual_evaluator(dataset, device)
    var_names = list(dataset.atmospheric_features)
    n_levels = len(dataset.pressure_levels)
    channels = [j * n_levels + level_idx for j in range(len(var_names))]
    dx_km = compute_dx_km(dataset.grid_lat)
    H = max(leads)
    lead_idx = [k - 1 for k in leads]

    indices = np.unique(np.linspace(0, len(dataset) - 1, min(n_inits, len(dataset))).round().astype(int))
    loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=inits_per_batch, shuffle=False,
                        num_workers=num_workers)

    sums = {f"{who}_{kind}": 0.0 for who in ("pred", "true") for kind in PSD_KINDS}
    n_pred = n_true = 0
    for batch in tqdm(loader, desc="PSD rollout"):
        clock = batch["target_clock"][:, :H].to(device)
        pred_states, pred_changes = rollout_ensemble(model, batch["cond"].to(device), clock, n_members, norm)
        pred_changes = norm.unnorm_change(pred_changes)
        true_states = batch["target_states"][:, :H].to(device)
        true_prev = torch.cat([batch["init_state"][:, None].to(device), true_states[:, :-1]], dim=1)

        fields = {
            "pred_state": pred_states[:, :, lead_idx][..., channels, :, :].flatten(0, 1),  # (B*M, L, V, lon, lat)
            "pred_change": pred_changes[:, :, lead_idx][..., channels, :, :].flatten(0, 1),
            "true_state": true_states[:, lead_idx][..., channels, :, :],                    # (B, L, V, lon, lat)
            "true_change": (true_states - true_prev)[:, lead_idx][..., channels, :, :],
        }
        for who, states in (("pred", pred_states[:, :, lead_idx].flatten(0, 1)), ("true", true_states[:, lead_idx])):
            fields[f"{who}_geo"], fields[f"{who}_ageo"] = wind_decomposition(residual_fn, states, n_levels, level_idx,
                                                                             var_names)
        for key, val in fields.items():
            k_centres, psd = radial_psd(val.cpu().numpy(), dx_km)
            sums[key] = sums[key] + psd.sum(0)
        n_pred += pred_states.shape[0] * pred_states.shape[1]
        n_true += true_states.shape[0]

    result = {key: val / (n_pred if key.startswith("pred") else n_true) for key, val in sums.items()}
    result.update(settings, wavelength_km=1.0 / k_centres[1:], var_names=np.array(var_names),
                  **{f"var_names_{kind}": np.array(["u", "v"]) for kind in WIND_KINDS},
                  level=str(int(dataset.pressure_levels[level_idx])),
                  init_times=np.array([dataset.init_time(k) for k in indices]))
    for key in sums:  # drop the DC bin, as for wavelength_km
        result[key] = result[key][..., 1:]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **result)
    print(f"Saved PSD ({len(indices)} inits x {n_members} members) to {out_path}")
    return result


def load_psd(path: Path) -> dict:
    with np.load(path) as saved:
        return {key: saved[key] for key in saved.files}


def plot_psd_grid(psd_by_model: dict[str, dict], kind: str, out_path: Path) -> Path:
    """One figure of mean PSDs (rows = leads, columns = variables) for one or
    more models' `compute_rollout_psd` results, with ERA5 truth in black.
    `kind` is "change" or "state". All models must share the same inits."""
    missing = [label for label, psd in psd_by_model.items() if f"pred_{kind}" not in psd]
    if missing:
        print(f"No '{kind}' PSD for {missing} (saved before it existed); leaving them out of this plot.")
    psd_by_model = {label: psd for label, psd in psd_by_model.items() if label not in missing}
    if not psd_by_model:
        return None
    labels = list(psd_by_model)
    ref = psd_by_model[labels[0]]
    for label in labels[1:]:
        if not np.array_equal(psd_by_model[label]["init_times"], ref["init_times"]):
            print(f"Warning: {label} PSD uses different inits than {labels[0]}; truth curve is from {labels[0]}.")

    leads = list(ref["leads"])
    var_names = [str(v) for v in ref.get(f"var_names_{kind}", ref["var_names"])]
    fig, axes = plt.subplots(len(leads), len(var_names), figsize=(4 * len(var_names), 3.3 * len(leads)),
                             squeeze=False)
    for i, lead in enumerate(leads):
        for j, var in enumerate(var_names):
            ax = axes[i, j]
            ax.loglog(ref["wavelength_km"], ref[f"true_{kind}"][i, j], color="black", lw=1.8, label="ERA5")
            for idx, label in enumerate(labels):
                psd = psd_by_model[label]
                ax.loglog(psd["wavelength_km"], psd[f"pred_{kind}"][i, j], color=get_model_color(idx, len(labels)),
                          lw=1.5, label=label)
            ax.invert_xaxis()  # large scales on the left, small scales (blurring) on the right
            ax.grid(True, which="both", alpha=0.3)
            ax.set_title(f"{var} ({ref['level']} hPa), lead {lead} h")
            if i == len(leads) - 1:
                ax.set_xlabel("Wavelength (km)")
            if j == 0:
                ax.set_ylabel("Power spectral density")

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=min(len(legend_labels), 5), bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"PSD of the {PSD_KINDS[kind]}: mean over {len(ref['init_times'])} inits x {int(ref['n_members'])} members")
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved PSD plot to {out_path}")
    return out_path
