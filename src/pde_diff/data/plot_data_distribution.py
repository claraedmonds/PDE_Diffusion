import argparse
import json
from pathlib import Path

import torch
from pde_diff.utils import DatasetRegistry, LossRegistry
from pde_diff.loss import VorticityLoss
from tqdm import tqdm
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os

font_path = "./times.ttf"
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'Times New Roman'

plt.style.use("pde_diff.custom_style")


def to_tensor(x, device):
    # Handles numpy arrays or already-a-tensor
    if isinstance(x, torch.Tensor):
        t = x
    else:
        t = torch.from_numpy(x)
    # Ensure floating point
    if not torch.is_floating_point(t):
        t = t.float()
    return t.to(device, non_blocking=True)


def append_hist(x, buf, max_samples: int):
    """
    Append flattened residual values to a Python list up to max_samples.
    Prevents unbounded RAM usage on large datasets.
    """
    x = x.detach().flatten().cpu()
    if len(buf) >= max_samples:
        return
    remaining = max_samples - len(buf)
    buf.extend(x[:remaining].tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config", type=Path,
                        default=Path("configs/dataset/era5_precomputed.yaml"))
    parser.add_argument("--out-json", type=Path,
                        default=Path("src/pde_diff/data/residual_stats.json"),
                        help="Where to save exact (uncapped) R1/R2 mean & std.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    plot_residual_hists = True
    plot_data_hists = False

    cfg = OmegaConf.load(args.dataset_config)

    # Cap number of samples to avoid excessive memory usage
    MAX_SAMPLES = 10_000

    if plot_residual_hists:
        cfg_loss = OmegaConf.create(
            {
                "name": "vorticity",
                "c_residual": 1.0,
            }
        )

        loss_fn = LossRegistry.create(cfg_loss)
        dataset = DatasetRegistry.create(cfg)
        loss_fn.set_mean_and_std(dataset.means, dataset.stds, dataset.diff_means, dataset.diff_stds)

        # Store raw (unnormalized) residual values for histograms
        hist_data = {
            "geostrophic_wind": [],
            "planetary_vorticity": [],
        }
        hist_data_noise = {
            "geostrophic_wind_noise": [],
            "planetary_vorticity_noise": [],
        }
        relative_wind_hist = {
            "geostrophic_wind_relative_error_u": [],
            "geostrophic_wind_relative_error_v": []
        }
        # Exact (uncapped) running mean/std, independent of the MAX_SAMPLES-capped hist_data
        residual_stats = {
            "geostrophic_wind": {"count": 0, "sum": 0.0, "sumsq": 0.0},
            "planetary_vorticity": {"count": 0, "sum": 0.0, "sumsq": 0.0},
        }

        with torch.no_grad():
            for data in tqdm(dataset, desc="Processing dataset"):
                num_vars = (data[0].shape[0] - 8) // 6
                prev_np, curr_np = data[0][None, num_vars*3+4:-4], data[1][None, :]

                prev = to_tensor(prev_np, device)
                curr = to_tensor(curr_np, device)
                prev_noise = torch.randn_like(prev)
                curr_noise = torch.randn_like(curr)

                # Compute residuals (NOT normalized)
                r_gw = loss_fn.compute_residual_geostrophic_wind(prev, curr, normalize=False)
                r_pv = loss_fn.compute_residual_planetary_vorticity(prev, curr, normalize=False)
                rel_gw = loss_fn.compute_residual_geostrophic_wind(prev, curr, normalize=False, relative=True)

                # Compute residuals with noise (NOT normalized)
                # use noisy prev/curr to inspect effect of noise
                r_gw_noise = loss_fn.compute_residual_geostrophic_wind(prev_noise, curr_noise, normalize=False)
                r_pv_noise = loss_fn.compute_residual_planetary_vorticity(prev_noise, curr_noise, normalize=False)

                for key, r in (("geostrophic_wind", r_gw), ("planetary_vorticity", r_pv)):
                    s = residual_stats[key]
                    s["count"] += r.numel()
                    s["sum"] += r.sum().item()
                    s["sumsq"] += (r ** 2).sum().item()

                append_hist(r_gw, hist_data["geostrophic_wind"], MAX_SAMPLES)
                append_hist(r_pv, hist_data["planetary_vorticity"], MAX_SAMPLES)

                append_hist(rel_gw[0],relative_wind_hist["geostrophic_wind_relative_error_u"],MAX_SAMPLES)
                append_hist(rel_gw[1],relative_wind_hist["geostrophic_wind_relative_error_v"],MAX_SAMPLES)

                append_hist(r_gw_noise, hist_data_noise["geostrophic_wind_noise"], MAX_SAMPLES)
                append_hist(r_pv_noise, hist_data_noise["planetary_vorticity_noise"], MAX_SAMPLES)

                del prev, curr, r_gw, r_pv, prev_noise, curr_noise, r_gw_noise, r_pv_noise

        results = {}
        for key, s in residual_stats.items():
            mean = s["sum"] / s["count"]
            var = s["sumsq"] / s["count"] - mean ** 2
            results[key] = {"mean": mean, "std": var ** 0.5}
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(results, indent=2))
        print(f"Saved residual statistics to {args.out_json}")

        # --- Plot 2 histograms side-by-side (counts, not normalized) ---
        fig, axes = plt.subplots(1, 2, figsize=(8, 2.5), sharey=True)

        plots = [
            ("planetary_vorticity", r"$\mathcal{R}_1$ Planetary Vorticity Residual"),
            ("geostrophic_wind", r"$\mathcal{R}_2$ Geostrophic Wind Residual"),
        ]

        for ax, (key, title) in zip(axes, plots):
            data = np.asarray(hist_data[key])

            mean = data.mean()
            std = data.std()
            col = "#1A4DAC"

            # Robust x-limits to avoid extreme tails dominating
            lo, hi = np.percentile(data, [0.5, 99.5])

            # Histogram
            ax.hist(
                data,
                bins=120,
                range=(lo, hi),
                histtype="stepfilled",
                alpha=0.6,
                edgecolor="black",
                linewidth=0.8,
                color=col,
            )

            # Mean and std lines
            ax.axvline(mean, linestyle="-", linewidth=2, label="Mean",color=col)
            ax.axvline(mean - std, linestyle="--", linewidth=1.5, label="±1 Std",color=col)
            ax.axvline(mean + std, linestyle="--", linewidth=1.5,color=col)

            # Annotation box
            textstr = (
                f"Mean = {mean:.3e}\n"
                f"Std  = {std:.3e}"
            )
            ax.text(
                0.97,
                0.97,
                textstr,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
            )

            ax.set_title(title, fontsize=12)
            ax.set_xlabel("Residual value")
            ax.set_xlim(lo, hi)

            ax.grid(True, linestyle=":", alpha=0.6)

        axes[0].set_ylabel("Count")

        plt.tight_layout()
        plt.savefig("reports/figures/era5_residual_histograms.png", dpi=300)
        print("Saved histogram figure to reports/figures/era5_residual_histograms.png")

        fig, axes = plt.subplots(1, 2, figsize=(8, 2.5), sharey=True)
        plots = [
            ("geostrophic_wind_relative_error_u", r"$\mathcal{R}_2$ Relative Residual, u component"),
            ("geostrophic_wind_relative_error_v", r"$\mathcal{R}_2$ Relative Residual, v component"),
        ]

        for ax, (key, title) in zip(axes, plots):
            data = np.asarray(relative_wind_hist[key])

            mean = data.mean()
            std = data.std()
            col = "#2A9D8F"
            
            # Robust x-limits to avoid extreme tails dominating
            lo, hi = np.percentile(data, [0.5, 99.5])
            mean = data[(data>lo)&(data<hi)].mean()
            std = data[(data>lo)&(data<hi)].std()

            # Histogram
            ax.hist(
                data,
                bins=120,
                range=(lo, hi),
                histtype="stepfilled",
                alpha=0.6,
                edgecolor="black",
                linewidth=0.8,
                color=col,
            )

            # Mean and std lines
            ax.axvline(mean, linestyle="-", linewidth=2, label="Mean",color=col)
            ax.axvline(mean - std, linestyle="--", linewidth=1.5, label="±1 Std",color=col)
            ax.axvline(mean + std, linestyle="--", linewidth=1.5,color=col)

            # Annotation box
            textstr = (
                f"Mean = {mean:.3e}\n"
                f"Std  = {std:.3e}"
            )
            ax.text(
                0.97,
                0.97,
                textstr,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
            )

            ax.set_title(title, fontsize=12)
            ax.set_xlabel("Residual value")
            ax.set_xlim(lo, hi)

            ax.grid(True, linestyle=":", alpha=0.6)

        axes[0].set_ylabel("Count")

        plt.tight_layout()
        plt.savefig("reports/figures/era5_relative_geo_wind_residual_histograms.png", dpi=300)
        print("Saved histogram figure to reports/figures/era5_relative_geo_wind_residual_histograms.png")

        # --- Plot 2 histograms side-by-side for noise (counts, not normalized) ---
        fig, axes = plt.subplots(1, 2, figsize=(8, 2.5), sharey=True)

        plots = [
            ("planetary_vorticity_noise", r"$\mathcal{R}_1$ Planetary Vorticity Residual on Noise"),
            ("geostrophic_wind_noise", r"$\mathcal{R}_2$ Geostrophic Wind Residual on Noise"),
        ]

        for ax, (key, title) in zip(axes, plots):
            data = np.asarray(hist_data_noise[key])

            mean = data.mean()
            std = data.std()
            col = "#E76F51"

            # Robust x-limits to avoid extreme tails dominating
            lo, hi = np.percentile(data, [0.5, 99.5])

            # Histogram
            ax.hist(
                data,
                bins=120,
                range=(lo, hi),
                histtype="stepfilled",
                alpha=0.6,
                edgecolor="black",
                linewidth=0.8,
                color=col
            )

            # Mean and std lines
            ax.axvline(mean, linestyle="-", linewidth=2, label="Mean",color=col)
            ax.axvline(mean - std, linestyle="--", linewidth=1.5, label="±1 Std",color=col)
            ax.axvline(mean + std, linestyle="--", linewidth=1.5,color=col)

            # Annotation box
            textstr = (
                f"Mean = {mean:.3e}\n"
                f"Std  = {std:.3e}"
            )
            ax.text(
                0.97,
                0.97,
                textstr,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
            )

            ax.set_title(title, fontsize=12)
            ax.set_xlabel("Residual value")
            ax.set_xlim(lo, hi)

            ax.grid(True, linestyle=":", alpha=0.6)

        axes[0].set_ylabel("Count")

        plt.tight_layout()
        plt.savefig("reports/figures/era5_residual_histograms_noise.png", dpi=300)
        print("Saved histogram figure to reports/figures/era5_residual_histograms_noise.png")

    if plot_data_hists:
        cfg.normalize = False
        dataset = DatasetRegistry.create(cfg)

        vars = ["u","v","pv","t","z"]
        levels = ["450hPa","500hPa","550hPa"]
        hist_data = {str(j): {str(k):[] for k,lvl in enumerate(levels)} for j,var in enumerate(vars)}

        from pde_diff.visualize import VAR_NAMES, VAR_UNITS
        colors = ["#2A9D8F", "#E76F51", "#1A4DAC"]

        
        for data in tqdm(dataset, desc="Processing dataset"):
            state1_np = data[0][None, :19]

            state1 = to_tensor(state1_np, device)

            for j,var in enumerate(vars):
                for k,lvl in enumerate(levels):
                    data_var = state1[0,(j*3+k)].flatten()
                    append_hist(data_var, hist_data[str(j)][str(k)], MAX_SAMPLES)

            del state1, data_var

        # --- Plot 5 variable state histograms ---
        
        for j,var in enumerate(vars):
            fig, ax = plt.subplots(figsize=(4.1,4.1))
            all_data=np.array([])
            for k,lvl in enumerate(levels):
                data = np.asarray(hist_data[str(j)][str(k)])
                all_data = np.concatenate((all_data,data))
                # Robust x-limits to avoid extreme tails dominating

            lo, hi = np.percentile(all_data, [0.5, 99.5])
            text_lines = []
            for k,lvl in enumerate(levels):

                data = np.asarray(hist_data[str(j)][str(k)])

                mean = data.mean()
                std = data.std()

                # Histogram
                ax.hist(
                    data,
                    bins=120,
                    range=(lo, hi),
                    histtype="stepfilled",
                    alpha=0.6,
                    color=colors[k],
                    edgecolor="black",
                    linewidth=0.8,
                    label=f"{lvl} Mean = {mean:.2e}, Std  = {std:.2e}"
                )
                mean = data.mean()
                std = data.std()

                # Mean and std lines
                ax.axvline(mean, color=colors[k], linestyle="-", linewidth=2)
                ax.axvline(mean - std, color=colors[k], linestyle="--", linewidth=1.5)
                ax.axvline(mean + std, color=colors[k], linestyle="--", linewidth=1.5)

            ax.set_title(f"Distribution of {VAR_NAMES[var]}", fontsize=12)
            ax.set_xlabel(f"{VAR_UNITS[var]}")
            ax.set_xlim(lo, hi)

            ax.grid(True, linestyle=":", alpha=0.6)

            ax.set_ylabel("Count")

            ax.legend()

            plt.tight_layout()
            dir = "reports/figures/histograms"
            os.makedirs(dir, exist_ok=True)
            plot_path =dir+f"/era5_{var}_histograms.png"
            plt.savefig(plot_path, dpi=300)
            print(f"Saved histogram figure to {plot_path}")

        
        hist_data = {str(j): {str(k):[] for k,lvl in enumerate(levels)} for j,var in enumerate(vars)}

        for data in tqdm(dataset, desc="Processing dataset"):
            change_np = data[1]

            change = to_tensor(change_np, device)

            for j,var in enumerate(vars):
                for k,lvl in enumerate(levels):
                    data_var = change[(j*3+k)].flatten()
                    append_hist(data_var, hist_data[str(j)][str(k)], MAX_SAMPLES)

            del change, data_var
        
        # --- Plot 5 change of state histograms  ---
        for j,var in enumerate(vars):
            fig, ax = plt.subplots(figsize=(4.1,4.1))
            all_data=np.array([])
            for k,lvl in enumerate(levels):
                data = np.asarray(hist_data[str(j)][str(k)])
                all_data = np.concatenate((all_data,data))
                # Robust x-limits to avoid extreme tails dominating

            lo, hi = np.percentile(all_data, [0.5, 99.5])
            text_lines = []
            for k,lvl in enumerate(levels):

                data = np.asarray(hist_data[str(j)][str(k)])

                mean = data.mean()
                std = data.std()

                # Histogram
                ax.hist(
                    data,
                    bins=120,
                    range=(lo, hi),
                    histtype="stepfilled",
                    alpha=0.6,
                    color=colors[k],
                    edgecolor="black",
                    linewidth=0.8,
                    label=f"{lvl} Mean = {mean:.2e}, Std  = {std:.2e}"
                )
                mean = data.mean()
                std = data.std()

                # Mean and std lines
                ax.axvline(mean, color=colors[k], linestyle="-", linewidth=2)
                ax.axvline(mean - std, color=colors[k], linestyle="--", linewidth=1.5)
                ax.axvline(mean + std, color=colors[k], linestyle="--", linewidth=1.5)

            ax.set_title(f"Distribution of change of {VAR_NAMES[var]}", fontsize=12)
            ax.set_xlabel(f"{VAR_UNITS[var]}")
            ax.set_xlim(lo, hi)

            ax.grid(True, linestyle=":", alpha=0.6)

            ax.set_ylabel("Count")
            ax.legend()

            plt.tight_layout()
            dir = "reports/figures/histograms"
            os.makedirs(dir, exist_ok=True)
            plot_path =dir+f"/era5_{var}_change_histograms.png"
            plt.savefig(plot_path, dpi=300)
            print(f"Saved histogram figure to {plot_path}")
