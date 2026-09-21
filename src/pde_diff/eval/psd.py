"""
Radial power-spectral-density (PSD) computation and plotting for comparing
ERA5 ground truth vs. one or more trained models' forecasts, at 500 hPa.

All variables are drawn onto a single figure (one subplot per variable)
rather than one PNG per variable.
"""
from pathlib import Path

import numpy as np
from scipy.fft import fft2, fftfreq
import matplotlib.pyplot as plt

from pde_diff.eval_primitives import PLOT_TYPE
from pde_diff.eval.palette import get_model_color

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
    the dataset. The ERA5 crop is a regular lat-lon grid, not equal-area, so
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


def plot_psd_comparison_on_axis(
    ax,
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    variable: str,
    model_labels: list[str] | None = None,
) -> None:
    """Draw one variable's PSD comparison (ground truth + every model in
    `curves`) onto a given Axes."""
    wl_gt, psd_gt = curves["ground_truth"]
    ax.loglog(wl_gt, psd_gt, color="black", lw=1.8, label="ERA5 (ground truth)")

    if model_labels is None:
        model_labels = [label for label in curves if label != "ground_truth"]
    n_models = len(model_labels)

    for model_idx, label in enumerate(model_labels):
        wl, psd = curves[label]
        color = get_model_color(model_idx, n_models)
        ax.loglog(wl, psd, color=color, lw=1.5, label=label)

    ax.set_xlabel("Wavelength (km)")
    ax.set_ylabel("Power spectral density")
    ax.set_title(f"{variable} (500 hPa)")
    ax.invert_xaxis()  # large scales on the left, small scales (blurring) on the right
    ax.grid(True, which="both", alpha=0.3)


def plot_all_psd_variables(
    ground_truth_state: np.ndarray,
    predictions_by_model: dict[str, np.ndarray],
    grid_lat: np.ndarray,
    var_names: list[str],
    out_dir: Path,
) -> Path:
    """
    ground_truth_state   : (var, lon, lat) physical-unit field at 500 hPa.
    predictions_by_model : {model_id: (var, lon, lat)}, same shape.

    Computes dx_km once from grid_lat and produces one figure with a subplot
    per variable (`len(var_names)` panels total), saved as a single PNG.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dx_km = compute_dx_km(grid_lat)
    model_labels = list(predictions_by_model.keys())

    n_vars = len(var_names)
    fig, axes = plt.subplots(1, n_vars, figsize=(5 * n_vars, 4.5))
    if n_vars == 1:
        axes = [axes]

    for j, var in enumerate(var_names):
        preds = {model_id: arr[j] for model_id, arr in predictions_by_model.items()}
        curves = compute_psd_curves(ground_truth_state[j], preds, dx_km)
        plot_psd_comparison_on_axis(axes[j], curves, var, model_labels=model_labels)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 5), bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Power spectral density")
    fig.tight_layout()

    path = out_dir / f"psd_all_variables{PLOT_TYPE}"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
