"""
Radial power-spectral-density (PSD) computation and plotting for comparing
ERA5 ground truth vs. one or more trained models' forecasts, at 500 hPa.

Called from evaluate.py's evaluate_forecasting() via plot_all_psd_variables().
"""
from pathlib import Path

import numpy as np
from scipy.fft import fft2, fftfreq

try:
    from pde_diff.visualize import PLOT_TYPE, pidm_colors, diffusion_colors, model_id_to_name
except Exception:
    PLOT_TYPE = ".png"
    pidm_colors = {}
    diffusion_colors = ("#8800FF", "#5900A7")
    model_id_to_name = {}

import matplotlib.pyplot as plt

EARTH_RADIUS_KM = 6371.0088
DEG_TO_KM = np.pi / 180.0 * EARTH_RADIUS_KM  # ~111.32 km/deg


def radial_psd(field: np.ndarray, dx_km: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the azimuthally-averaged (radial) power spectral density of a 2-D field.

    A Hann window is applied before the FFT. Without it, the field's sharp
    rectangular edges act as an implicit boxcar window, whose FFT leaks
    broadband power into every wavenumber bin (spectral leakage) — this masks
    real high-wavenumber differences between fields (e.g. it can make a
    heavily-blurred field's PSD look nearly identical to the unblurred one).

    No land-crop/region argument: unlike the reference implementation this is
    adapted from, the ERA5 crop used in this repo has no land to avoid, so the
    full field passed in is used directly.

    Parameters
    ----------
    field  : 2-D array.
    dx_km  : grid spacing [km] (isotropic — see compute_dx_km).

    Returns
    -------
    k_centres : np.ndarray — wavenumber bin centres [cycles / km]
    psd       : np.ndarray — mean power in each bin
    """
    if np.isnan(field).any():
        field = field.copy()
        field[np.isnan(field)] = np.nanmean(field)

    ny, nx = field.shape

    window = np.hanning(ny)[:, None] * np.hanning(nx)[None, :]
    window /= np.sqrt(np.mean(window ** 2))  # preserve overall power scale

    f2d = fft2((field - field.mean()) * window)
    power = (np.abs(f2d) ** 2) / (ny * nx)

    ky = fftfreq(ny, d=dx_km)
    kx = fftfreq(nx, d=dx_km)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX ** 2 + KY ** 2)

    k_max = 0.5 / dx_km
    n_bins = min(ny, nx) // 2
    k_bins = np.linspace(0, k_max, n_bins + 1)
    psd = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (K >= k_bins[i]) & (K < k_bins[i + 1])
        if mask.any():
            psd[i] = power[mask].mean()
            counts[i] = mask.sum()
    k_centres = 0.5 * (k_bins[:-1] + k_bins[1:])
    valid = counts > 0
    return k_centres[valid], psd[valid]


def radial_psd_wavelength(field: np.ndarray, dx_km: float) -> tuple[np.ndarray, np.ndarray]:
    """Radially averaged 2-D PSD. Returns (wavelength_km, power), DC bin dropped."""
    k, psd = radial_psd(field, dx_km)
    k, psd = k[1:], psd[1:]
    return 1.0 / k, psd


def compute_dx_km(grid_lat: np.ndarray) -> float:
    """
    Isotropic grid spacing [km] derived from the actual latitude spacing of
    the dataset (not hardcoded, so it stays correct if downsample_factor
    changes). The ERA5 crop is a regular lat-lon grid, not equal-area, so
    longitude spacing shrinks with cos(latitude); using the (constant)
    latitude-direction spacing for both FFT axes is an approximation. This
    only affects the absolute km/wavelength calibration on the x-axis — since
    the same dx_km is applied identically to ground truth and every model
    being compared, it does not distort the relative comparison between them.
    """
    grid_spacing_deg = float(np.abs(np.diff(np.sort(np.unique(grid_lat)))).mean())
    return grid_spacing_deg * DEG_TO_KM


def compute_psd_curves(
    ground_truth: np.ndarray,
    predictions: dict[str, np.ndarray],
    dx_km: float,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    ground_truth : (lon, lat) physical-unit field.
    predictions  : {model_label: (lon, lat) field}, same shape as ground_truth.

    Returns {"ground_truth": (wavelength_km, psd), model_label: (wavelength_km, psd), ...}
    """
    curves = {"ground_truth": radial_psd_wavelength(ground_truth, dx_km)}
    for label, field in predictions.items():
        curves[label] = radial_psd_wavelength(field, dx_km)
    return curves


def _color_for_model(label: str, model_idx: int) -> str:
    if label in pidm_colors:
        return pidm_colors[label]
    if model_idx == 0:
        return diffusion_colors[0]
    tab10 = plt.get_cmap("tab10")
    return tab10(model_idx % 10)


def plot_psd_comparison(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    variable: str,
    out_dir: Path,
) -> Path:
    """One log-log PSD figure overlaying ground truth (black) and every model
    in `curves` (colors from the existing repo palettes where recognized,
    otherwise a tab10 fallback cycle). Saves to out_dir/psd_<variable>.png."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4.5))

    wl_gt, psd_gt = curves["ground_truth"]
    ax.loglog(wl_gt, psd_gt, color="black", lw=1.8, label="ERA5 (ground truth)")

    model_idx = 0
    for label, (wl, psd) in curves.items():
        if label == "ground_truth":
            continue
        color = _color_for_model(label, model_idx)
        legend_label = model_id_to_name.get(label, label)
        ax.loglog(wl, psd, color=color, lw=1.5, label=legend_label)
        model_idx += 1

    ax.set_xlabel("Wavelength (km)")
    ax.set_ylabel("Power spectral density")
    ax.set_title(f"PSD — {variable} (500 hPa)")
    ax.invert_xaxis()  # large scales on the left, small scales (blurring) on the right
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    path = out_dir / f"psd_{variable}{PLOT_TYPE}"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_all_psd_variables(
    ground_truth_state: np.ndarray,
    predictions_by_model: dict[str, np.ndarray],
    grid_lat: np.ndarray,
    var_names: list[str],
    out_dir: Path,
) -> list[Path]:
    """
    ground_truth_state   : (var, lon, lat) physical-unit field at 500 hPa.
    predictions_by_model : {model_id: (var, lon, lat)}, same shape.

    Computes dx_km once from grid_lat and produces one PSD comparison PNG per
    variable (len(var_names) files total).
    """
    dx_km = compute_dx_km(grid_lat)
    paths = []
    for j, var in enumerate(var_names):
        preds = {model_id: arr[j] for model_id, arr in predictions_by_model.items()}
        curves = compute_psd_curves(ground_truth_state[j], preds, dx_km)
        paths.append(plot_psd_comparison(curves, var, out_dir))
    return paths
