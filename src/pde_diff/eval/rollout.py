"""Autoregressive ensemble rollout evaluation on the hourly test trajectory
(TrajectoryDataset): every init is rolled forward `horizon` hours with
`n_members` independent diffusion samples, and pluggable metrics are written
as long-format rows (init_time, lead_h, metric, variable, level, member, value)
to `rollout_metrics*.csv`.

Adding a metric: decorate a `fn(ctx: RolloutContext) -> list[dict]` with
`@register_metric("<name>")` and list `<name>` in `eval.rollout.metrics`.
Because every row carries its init time, event-based evaluation (e.g. extreme
events) can filter the same rows by date or select inits via
`eval.rollout.init_times` without touching the rollout engine.
"""
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from pde_diff.eval.residuals import build_residual_evaluator
from pde_diff.eval.samples import (
    plot_forecasts_vs_targets,
    plot_residuals_with_truth,
    plot_recursive_residuals,
    plot_mse_of_all_vars,
    plot_mse_of_vars,
    plot_forecast_error_distributions,
)

METRICS_FILE_STEM = "rollout_metrics"
COLUMNS = ["init_time", "lead_h", "metric", "variable", "level", "member", "value"]
DEFAULT_METRICS = ["mse", "residuals", "spread", "crps", "persistence", "rmse"]
QUALITATIVE_LEVEL_IDX = 1  # 500 hPa

# (row metric name, VorticityLoss method, short name used by the residual plotters)
RESIDUALS = [
    ("R1_planetary_vorticity", "compute_residual_planetary_vorticity", "plan"),
    ("R2_geostrophic_wind", "compute_residual_geostrophic_wind", "gw"),
    ("R3_vorticity_divergence", "compute_residual_vorticity_divergence", "vd"),
]


class Normalizer:
    """Per-channel state / state-change (un)normalisation for tensors whose
    channel axis is -3, using the training normalisation stats."""

    def __init__(self, dataset, device):
        def as_t(a):
            return torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)[:, None, None]
        self.means, self.stds = as_t(dataset.means), as_t(dataset.stds)
        self.diff_means, self.diff_stds = as_t(dataset.diff_means), as_t(dataset.diff_stds)

    def norm_state(self, x):
        return (x - self.means) / self.stds

    def unnorm_state(self, x):
        return x * self.stds + self.means

    def norm_change(self, x):
        return (x - self.diff_means) / self.diff_stds

    def unnorm_change(self, x):
        return x * self.diff_stds + self.diff_means


@torch.no_grad()
def rollout_ensemble(model, cond, target_clock, n_members: int, norm: Normalizer):
    """Roll `model` forward autoregressively from `cond` (B, 2*(C+4), lon, lat),
    drawing `n_members` independent samples per init. The clock features of
    each new conditioning step are the true ones for that lead
    (`target_clock`, (B, H, 4, lon, lat)), so no clock drift accumulates.

    Returns (raw predicted states, normalised predicted changes), both
    shaped (B, M, H, C, lon, lat)."""
    B, H = target_clock.shape[:2]
    per_time = cond.shape[1] // 2
    C = per_time - target_clock.shape[2]

    x = cond.repeat_interleave(n_members, dim=0)
    clock = target_clock.repeat_interleave(n_members, dim=0)
    state = norm.unnorm_state(x[:, per_time:per_time + C])

    states, changes = [], []
    for k in range(H):
        change = model.sample_loop(batch_size=x.size(0), conditionals=x)
        state = state + norm.unnorm_change(change)
        x = torch.cat([x[:, per_time:], norm.norm_state(state), clock[:, k]], dim=1)
        states.append(state)
        changes.append(change)

    shape = (B, n_members, H, *state.shape[1:])
    return torch.stack(states, dim=1).reshape(shape), torch.stack(changes, dim=1).reshape(shape)


def residual_fields(residual_fn, prev_norm, change_norm) -> dict[str, torch.Tensor]:
    """|R1|, |R2|, |R3| fields (normalised residual units) for a batch of
    normalised previous states and normalised changes, each (N, C, lon, lat)."""
    return {
        name: getattr(residual_fn, method)(x0_previous=prev_norm, x0_change_pred=change_norm, normalize=True).abs()
        for name, method, _ in RESIDUALS
    }


@dataclass
class RolloutContext:
    """Everything a rollout metric may need for one batch of inits. States
    are raw (unnormalised); the H axis holds leads 1..H."""
    init_times: list[str]
    pred_states: torch.Tensor   # (B, M, H, C, lon, lat) raw
    pred_changes: torch.Tensor  # (B, M, H, C, lon, lat) normalised
    true_states: torch.Tensor   # (B, H, C, lon, lat) raw
    init_state: torch.Tensor    # (B, C, lon, lat) raw state at lead 0
    norm: Normalizer
    channel_names: list[tuple[str, str]]  # (variable, level) per channel
    lat_weights: torch.Tensor   # (lat,) cos(lat) weights with mean 1
    residual_fn: object         # VorticityLoss used as residual evaluator

    @property
    def n_members(self) -> int:
        return self.pred_states.shape[1]

    @cached_property
    def pred_states_norm(self):
        return self.norm.norm_state(self.pred_states)

    @cached_property
    def true_states_norm(self):
        return self.norm.norm_state(self.true_states)

    @cached_property
    def true_prev_states(self):
        """Raw true state at lead k-1 for each lead k: (B, H, C, lon, lat)."""
        return torch.cat([self.init_state[:, None], self.true_states[:, :-1]], dim=1)

    @cached_property
    def pred_prev_states(self):
        """Raw predicted state at lead k-1 for each lead k: (B, M, H, C, lon, lat)."""
        init = self.init_state[:, None, None].expand(-1, self.n_members, 1, -1, -1, -1)
        return torch.cat([init, self.pred_states[:, :, :-1]], dim=2)

    @cached_property
    def true_changes_norm(self):
        return self.norm.norm_change(self.true_states - self.true_prev_states)

    def rows(self, metric, member, per_channel=None, overall=None, variable="all", level="all") -> list[dict]:
        """Long-format rows from per-channel values (B, H, C) and/or an
        overall value (B, H)."""
        out = []
        if per_channel is not None:
            vals = per_channel.detach().cpu().numpy()
            for b, init in enumerate(self.init_times):
                for k in range(vals.shape[1]):
                    for c, (var, lev) in enumerate(self.channel_names):
                        out.append(dict(init_time=init, lead_h=k + 1, metric=metric, variable=var, level=lev,
                                        member=member, value=float(vals[b, k, c])))
        if overall is not None:
            vals = overall.detach().cpu().numpy()
            for b, init in enumerate(self.init_times):
                for k in range(vals.shape[1]):
                    out.append(dict(init_time=init, lead_h=k + 1, metric=metric, variable=variable, level=level,
                                    member=member, value=float(vals[b, k])))
        return out


ROLLOUT_METRICS: dict[str, Callable[[RolloutContext], list[dict]]] = {}


def register_metric(name: str):
    def decorator(fn):
        ROLLOUT_METRICS[name] = fn
        return fn
    return decorator


def _grid_mean(x):
    return x.mean(dim=(-2, -1))


@register_metric("mse")
def mse_metric(ctx: RolloutContext) -> list[dict]:
    """MSE of normalised states (per member, member mean, ensemble mean) and
    of normalised state changes (member mean)."""
    per_member = _grid_mean((ctx.pred_states_norm - ctx.true_states_norm[:, None]) ** 2)  # (B, M, H, C)
    rows = []
    for m in range(ctx.n_members):
        rows += ctx.rows("mse", str(m), overall=per_member[:, m].mean(-1))
    member_mean = per_member.mean(1)
    rows += ctx.rows("mse", "member_mean", per_channel=member_mean, overall=member_mean.mean(-1))
    ens = _grid_mean((ctx.pred_states_norm.mean(1) - ctx.true_states_norm) ** 2)
    rows += ctx.rows("mse", "ens_mean", per_channel=ens, overall=ens.mean(-1))
    change = _grid_mean((ctx.pred_changes - ctx.true_changes_norm[:, None]) ** 2).mean(1)
    rows += ctx.rows("mse_change", "member_mean", per_channel=change, overall=change.mean(-1))
    return rows


@register_metric("residuals")
def residual_metric(ctx: RolloutContext) -> list[dict]:
    """Mean |R1|, |R2|, |R3| per lead on the forecasts (per member and member
    mean) and on the ERA5 truth ("truth"), the reference for how far the real
    atmosphere itself is from satisfying each equation."""
    B, M, H = ctx.pred_states.shape[:3]
    pred_prev = ctx.norm.norm_state(ctx.pred_prev_states)
    true_prev = ctx.norm.norm_state(ctx.true_prev_states)
    pred_vals = {name: torch.zeros(B, M, H) for name, _, _ in RESIDUALS}
    true_vals = {name: torch.zeros(B, H) for name, _, _ in RESIDUALS}
    for k in range(H):
        pred = residual_fields(ctx.residual_fn, pred_prev[:, :, k].flatten(0, 1), ctx.pred_changes[:, :, k].flatten(0, 1))
        true = residual_fields(ctx.residual_fn, true_prev[:, k], ctx.true_changes_norm[:, k])
        for name in pred:
            pred_vals[name][:, :, k] = pred[name].flatten(1).mean(1).view(B, M).cpu()
            true_vals[name][:, k] = true[name].flatten(1).mean(1).cpu()

    rows = []
    for name, _, _ in RESIDUALS:
        for m in range(M):
            rows += ctx.rows(name, str(m), overall=pred_vals[name][:, m])
        rows += ctx.rows(name, "member_mean", overall=pred_vals[name].mean(1))
        rows += ctx.rows(name, "truth", overall=true_vals[name])
    return rows


@register_metric("spread")
def spread_metric(ctx: RolloutContext) -> list[dict]:
    """Ensemble spread, sqrt((M+1)/M * mean member variance), next to the
    ensemble-mean RMSE (normalised units); equal values mean a calibrated
    ensemble."""
    M = ctx.n_members
    if M < 2:
        return []
    var = _grid_mean(ctx.pred_states_norm.var(dim=1, unbiased=True)) * (M + 1) / M  # (B, H, C)
    ens_mse = _grid_mean((ctx.pred_states_norm.mean(1) - ctx.true_states_norm) ** 2)
    rows = ctx.rows("spread", "ens", per_channel=var.sqrt(), overall=var.mean(-1).sqrt())
    rows += ctx.rows("ens_mean_rmse", "ens_mean", per_channel=ens_mse.sqrt(), overall=ens_mse.mean(-1).sqrt())
    return rows


@register_metric("crps")
def crps_metric(ctx: RolloutContext) -> list[dict]:
    """Fair (unbiased) CRPS of the ensemble on normalised states:
    mean_i |x_i - y| - sum_{i,j} |x_i - x_j| / (2 M (M - 1))."""
    M = ctx.n_members
    x = ctx.pred_states_norm
    crps = (x - ctx.true_states_norm[:, None]).abs().mean(1)
    if M > 1:
        x_sorted, _ = x.sort(dim=1)
        rank = torch.arange(1, M + 1, device=x.device, dtype=x.dtype).view(1, M, 1, 1, 1, 1)
        pairwise_sum = 2 * ((2 * rank - M - 1) * x_sorted).sum(1)  # sum_{i,j} |x_i - x_j|
        crps = crps - pairwise_sum / (2 * M * (M - 1))
    crps = _grid_mean(crps)
    return ctx.rows("crps", "ens", per_channel=crps, overall=crps.mean(-1))


@register_metric("persistence")
def persistence_metric(ctx: RolloutContext) -> list[dict]:
    """Persistence baseline (state at lead 0 held fixed): normalised MSE and
    lat-weighted RMSE in physical units."""
    mse = _grid_mean((ctx.norm.norm_state(ctx.init_state)[:, None] - ctx.true_states_norm) ** 2)
    rmse = _grid_mean((ctx.init_state[:, None] - ctx.true_states) ** 2 * ctx.lat_weights).sqrt()
    return (ctx.rows("mse", "persistence", per_channel=mse, overall=mse.mean(-1))
            + ctx.rows("rmse", "persistence", per_channel=rmse))


@register_metric("rmse")
def rmse_metric(ctx: RolloutContext) -> list[dict]:
    """cos(lat)-weighted RMSE in physical units, per variable and level."""
    w = ctx.lat_weights
    ens = _grid_mean((ctx.pred_states.mean(1) - ctx.true_states) ** 2 * w).sqrt()
    members = _grid_mean((ctx.pred_states - ctx.true_states[:, None]) ** 2 * w).sqrt().mean(1)
    return ctx.rows("rmse", "ens_mean", per_channel=ens) + ctx.rows("rmse", "member_mean", per_channel=members)


class _Lead1ErrorMaps:
    """Running mean (over inits and members) of the squared lead-1 error of
    the normalised state change per variable at one level. Saved after every
    batch so a resumed run keeps accumulating."""

    def __init__(self, path: Path, n_vars: int, n_levels: int, level_idx: int):
        self.path, self.n_vars, self.n_levels, self.level_idx = path, n_vars, n_levels, level_idx
        if path.exists():
            saved = np.load(path)
            self.sum, self.count = saved["sum"], int(saved["count"])
        else:
            self.sum, self.count = None, 0

    def update(self, ctx: RolloutContext):
        channels = [j * self.n_levels + self.level_idx for j in range(self.n_vars)]
        sq = (ctx.pred_changes[:, :, 0, channels] - ctx.true_changes_norm[:, None, 0, channels]) ** 2
        batch_sum = sq.sum(dim=(0, 1)).cpu().numpy()
        self.sum = batch_sum if self.sum is None else self.sum + batch_sum
        self.count += sq.shape[0] * sq.shape[1]
        np.savez(self.path, sum=self.sum, count=self.count)

    def plot(self, out_dir: Path, var_names: list[str]):
        if self.sum is None:
            return
        maps = self.sum / self.count
        for j, var in enumerate(var_names):
            plot_mse_of_vars(maps[j], out_dir=out_dir, var=var)
        plot_mse_of_all_vars(maps, out_dir=out_dir, vars=var_names)


class _Lead1ErrorDistributions:
    """Collects lead-1 normalised state-change forecasts/targets (member 0)
    for the first `max_inits` processed inits, for error histograms."""

    def __init__(self, var_names: list[str], n_levels: int, max_inits: int):
        self.var_names, self.n_levels, self.max_inits = var_names, n_levels, max_inits
        self.preds = {v: [[] for _ in range(n_levels)] for v in var_names}
        self.targets = {v: [[] for _ in range(n_levels)] for v in var_names}
        self.n_inits = 0

    def update(self, ctx: RolloutContext):
        for b in range(len(ctx.init_times)):
            if self.n_inits >= self.max_inits:
                return
            pred = ctx.pred_changes[b, 0, 0].cpu().numpy()
            true = ctx.true_changes_norm[b, 0].cpu().numpy()
            for j, var in enumerate(self.var_names):
                for lvl in range(self.n_levels):
                    self.preds[var][lvl].append(pred[j * self.n_levels + lvl].ravel())
                    self.targets[var][lvl].append(true[j * self.n_levels + lvl].ravel())
            self.n_inits += 1

    def plot(self, out_dir: Path):
        if self.n_inits == 0:
            return
        for var in self.var_names:
            plot_forecast_error_distributions(self.preds[var], self.targets[var], variable=var, dir=out_dir)


def plot_qualitative(ctx: RolloutContext, b: int, sample_idx: int, out_dir: Path, var_names: list[str], n_levels: int,
                     level_idx: int = QUALITATIVE_LEVEL_IDX) -> None:
    """Forecast-vs-truth state/change grids and residual maps for member 0 of
    init `b` in the batch, at pressure level `level_idx`."""
    sample_dir = Path(out_dir) / f"rollout_{ctx.init_times[b]}"
    pred_states, true_states = ctx.pred_states[b, 0].cpu(), ctx.true_states[b].cpu()
    pred_changes = ctx.norm.unnorm_change(ctx.pred_changes[b, 0]).cpu()
    true_changes = (ctx.true_states[b] - ctx.true_prev_states[b]).cpu()
    H = pred_states.shape[0]

    for j, var in enumerate(var_names):
        ch = j * n_levels + level_idx
        plot_forecasts_vs_targets(forecasts=list(pred_states[:, ch]), targets=list(true_states[:, ch]),
                                  variable=var, sample_idx=sample_idx, dir=sample_dir)
        plot_forecasts_vs_targets(forecasts=list(pred_changes[:, ch]), targets=list(true_changes[:, ch]),
                                  variable=var, sample_idx=sample_idx, dir=sample_dir, states=False)

    pred = residual_fields(ctx.residual_fn, ctx.norm.norm_state(ctx.pred_prev_states[b, 0]), ctx.pred_changes[b, 0])
    true = residual_fields(ctx.residual_fn, ctx.norm.norm_state(ctx.true_prev_states[b]), ctx.true_changes_norm[b])
    for name, _, short in RESIDUALS:
        p, t = pred[name].cpu(), true[name].cpu()
        if p.dim() == 4:  # (H, lev, lon, lat): R2, R3 are per level
            p, t = p[:, level_idx], t[:, level_idx]
        plot_residuals_with_truth(p[0], t[0], short, sample_idx=sample_idx, dir=sample_dir)
        if H > 1:
            plot_recursive_residuals(list(p), list(t), short, sample_idx=sample_idx, dir=sample_dir)


def metrics_csv_path(out_dir: Path, shard: tuple[int, int] | None = None) -> Path:
    suffix = "" if shard is None else f"_shard{shard[0]}of{shard[1]}"
    return Path(out_dir) / f"{METRICS_FILE_STEM}{suffix}.csv"


def load_rollout_metrics(out_dir: Path) -> pd.DataFrame | None:
    """Concatenate all (possibly sharded) rollout metric CSVs in `out_dir`."""
    paths = sorted(Path(out_dir).glob(f"{METRICS_FILE_STEM}*.csv"))
    if not paths:
        return None
    dtypes = {"init_time": str, "metric": str, "variable": str, "level": str, "member": str}
    return pd.concat([pd.read_csv(p, dtype=dtypes) for p in paths], ignore_index=True)


def evaluate_rollout(
    model,
    dataset,
    out_dir: Path,
    n_members: int = 9,
    metrics: list[str] = DEFAULT_METRICS,
    inits_per_batch: int = 1,
    shard: tuple[int, int] | None = None,
    n_qualitative_samples: int = 1,
    plot_sample_grids: bool = True,
    n_error_dist_inits: int = 10,
    num_workers: int = 4,
) -> pd.DataFrame | None:
    """Roll `model` out from every init of `dataset` (a TrajectoryDataset),
    appending metric rows per batch of inits to `rollout_metrics*.csv` in
    `out_dir`. Inits already in the CSV are skipped, so an interrupted run
    resumes where it stopped; `shard=(i, n)` processes every n-th init from i.
    Returns all rollout metric rows found in `out_dir`."""
    unknown = [m for m in metrics if m not in ROLLOUT_METRICS]
    if unknown:
        raise ValueError(f"Unknown rollout metrics {unknown}; available: {sorted(ROLLOUT_METRICS)}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = metrics_csv_path(out_dir, shard)
    done = set(pd.read_csv(csv_path, usecols=["init_time"], dtype=str)["init_time"]) if csv_path.exists() else set()

    indices = list(range(len(dataset)))
    if shard is not None:
        indices = indices[shard[0]::shard[1]]
    todo = [k for k in indices if dataset.init_time(k) not in done]
    print(f"Rollout: {len(todo)} inits to run, {len(indices) - len(todo)} already in {csv_path.name}")

    device = model.device
    norm = Normalizer(dataset, device)
    residual_fn = build_residual_evaluator(dataset, device)
    var_names = list(dataset.atmospheric_features)
    levels = [str(int(p)) for p in dataset.pressure_levels]
    channel_names = [(var, lev) for var in var_names for lev in levels]
    lat_weights = torch.cos(torch.deg2rad(torch.as_tensor(dataset.grid_lat, dtype=torch.float32, device=device)))
    lat_weights = lat_weights / lat_weights.mean()

    shard_suffix = "" if shard is None else f"_shard{shard[0]}of{shard[1]}"
    error_maps = _Lead1ErrorMaps(out_dir / f"lead1_change_sq_err{shard_suffix}.npz", len(var_names), len(levels),
                                 QUALITATIVE_LEVEL_IDX)
    error_dists = _Lead1ErrorDistributions(var_names, len(levels), n_error_dist_inits)

    loader = DataLoader(Subset(dataset, todo), batch_size=inits_per_batch, shuffle=False, num_workers=num_workers)
    for batch in tqdm(loader, desc="Rollout"):
        cond = batch["cond"].to(device)
        pred_states, pred_changes = rollout_ensemble(model, cond, batch["target_clock"].to(device), n_members, norm)
        ctx = RolloutContext(
            init_times=list(batch["init_time"]),
            pred_states=pred_states,
            pred_changes=pred_changes,
            true_states=batch["target_states"].to(device),
            init_state=batch["init_state"].to(device),
            norm=norm,
            channel_names=channel_names,
            lat_weights=lat_weights,
            residual_fn=residual_fn,
        )
        with torch.no_grad():
            rows = [row for name in metrics for row in ROLLOUT_METRICS[name](ctx)]
            pd.DataFrame(rows, columns=COLUMNS).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
            error_maps.update(ctx)
            error_dists.update(ctx)
            if plot_sample_grids:
                for b, idx in enumerate(batch["index"].tolist()):
                    if idx < n_qualitative_samples:
                        plot_qualitative(ctx, b, idx, out_dir / "samples", var_names, len(levels))

    error_maps.plot(out_dir / "mse_maps", var_names)
    error_dists.plot(out_dir / "forecast_error_distributions")
    return load_rollout_metrics(out_dir)
