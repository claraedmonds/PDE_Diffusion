"""Tests for the rollout test set (TrajectoryDataset, init selection) and the
autoregressive rollout engine, on a small synthetic trajectory store."""
import json

import numpy as np
import torch
from omegaconf import OmegaConf

from pde_diff.data.datasets import TrajectoryDataset, select_init_indices
from pde_diff.eval.rollout import Normalizer, crps_metric, evaluate_rollout, load_rollout_metrics, rollout_ensemble
from pde_diff.utils import init_means_and_stds_era5

VARS = ["u", "v", "pv", "t", "z"]
LEVELS = [450.0, 500.0, 550.0]
N_LON, N_LAT = 480, 32  # VorticityLoss hardcodes this grid


def _hourly(start, n):
    return np.datetime64(start, "h") + np.arange(n).astype("timedelta64[h]")


def _segments(times):
    return np.concatenate([[0], np.cumsum(np.diff(times).astype(int) > 1)])


def _write_store(path, times, seed=0):
    """Synthetic trajectory store: smooth-in-time states around the ERA5
    normalisation means, with the training stats in metadata.json."""
    rng = np.random.default_rng(seed)
    means, stds, diff_means, diff_stds = init_means_and_stds_era5(VARS, [], [])
    T = len(times)
    base = rng.standard_normal((1, 15, N_LON, N_LAT)).astype(np.float32)
    drift = 0.05 * rng.standard_normal((T, 15, N_LON, N_LAT)).cumsum(0).astype(np.float32)
    states = (means[None, :, None, None] + stds[None, :, None, None] * (base + drift)).astype(np.float32)
    clock = rng.uniform(-1, 1, (T, 4, N_LON, N_LAT)).astype(np.float32)
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "states.npy", states)
    np.save(path / "clock.npy", clock)
    np.save(path / "times.npy", times)
    np.save(path / "segment_id.npy", _segments(times))
    meta = {
        "atmospheric_features": VARS,
        "pressure_levels": LEVELS,
        "grid_lon": np.linspace(0, 359, N_LON).tolist(),
        "grid_lat": np.linspace(70, 46, N_LAT).tolist(),
        "means": means.tolist(), "stds": stds.tolist(),
        "diff_means": diff_means.tolist(), "diff_stds": diff_stds.tolist(),
    }
    (path / "metadata.json").write_text(json.dumps(meta))
    return states


class OracleModel:
    """Stand-in for DiffusionModel whose `sample_loop` returns the true
    normalised change for whichever trajectory hour the conditioning holds."""

    def __init__(self, dataset):
        self.device = torch.device("cpu")
        self.dataset = dataset
        self.norm_states = torch.from_numpy(
            dataset._normalize(np.asarray(dataset.states), dataset.means, dataset.stds).astype(np.float32))

    def sample_loop(self, batch_size, conditionals):
        per_time = conditionals.shape[1] // 2
        current = conditionals[:, per_time:per_time + 15]
        out = []
        for x in current:
            t = int((self.norm_states - x).abs().mean(dim=(1, 2, 3)).argmin())
            change = self.dataset.states[t + 1] - self.dataset.states[t]
            out.append(torch.from_numpy(self.dataset._normalize(change, self.dataset.diff_means, self.dataset.diff_stds)))
        return torch.stack(out).float()


def test_select_init_indices_respects_segments_and_stride():
    # Jan-Feb 2023 | Dec 2023 + Jan 2024 (one DJF segment across the year boundary)
    times = np.concatenate([_hourly("2023-01-01T00", 24 * 59), _hourly("2023-12-01T00", 24 * 62)])
    seg = _segments(times)
    assert seg.max() == 1, "Dec 2023 -> Jan 2024 must be one segment"

    idx = select_init_indices(times, seg, horizon=12, stride=36)
    hours = (times[idx] - times[idx].astype("datetime64[D]")).astype(int)
    assert set(hours) == {0, 12}, "36 h stride must alternate 00/12 UTC starts"
    for t in idx:
        assert seg[t - 1] == seg[t] == seg[t + 12]
    assert any(times[t] >= np.datetime64("2024-01-01T00") for t in idx), "inits must continue into Jan 2024"
    assert np.all(np.diff(times[idx][seg[idx] == 1]).astype(int) == 36)

    explicit = select_init_indices(times, seg, horizon=12, explicit_times=["2023-01-15T06"])
    assert times[explicit][0] == np.datetime64("2023-01-15T06")


def test_trajectory_dataset_layout(tmp_path):
    times = _hourly("2023-01-01T00", 60)
    states = _write_store(tmp_path, times)
    ds = TrajectoryDataset(OmegaConf.create({"path": str(tmp_path), "horizon": 6, "init_stride": 12}))
    item = ds[0]
    t = int(ds.init_indices[0])
    assert item["init_time"] == str(times[t])
    norm = lambda x: (x - ds.means[:, None, None]) / ds.stds[:, None, None]  # noqa: E731
    cond = item["cond"].numpy()
    assert cond.shape == (38, N_LON, N_LAT)
    np.testing.assert_allclose(cond[:15], norm(states[t - 1]), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(cond[15:19], ds.clock[t - 1])
    np.testing.assert_allclose(cond[19:34], norm(states[t]), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(cond[34:38], ds.clock[t])
    np.testing.assert_array_equal(item["target_states"].numpy(), states[t + 1:t + 7])
    np.testing.assert_array_equal(item["target_clock"].numpy(), ds.clock[t + 1:t + 7])


def test_crps_matches_bruteforce():
    class Ctx:
        pass
    rng = torch.Generator().manual_seed(0)
    M = 5
    ctx = Ctx()
    ctx.n_members = M
    ctx.pred_states_norm = torch.randn(1, M, 2, 3, 4, 4, generator=rng)
    ctx.true_states_norm = torch.randn(1, 2, 3, 4, 4, generator=rng)
    captured = {}
    ctx.rows = lambda metric, member, per_channel=None, overall=None: captured.update(pc=per_channel) or []
    crps_metric(ctx)

    x, y = ctx.pred_states_norm, ctx.true_states_norm[:, None]
    pair = (x[:, :, None] - x[:, None]).abs().sum(dim=(1, 2))
    expected = ((x - y).abs().mean(1) - pair / (2 * M * (M - 1))).mean(dim=(-2, -1))
    torch.testing.assert_close(captured["pc"], expected)


def test_oracle_rollout_reproduces_truth(tmp_path):
    times = _hourly("2023-01-01T00", 40)
    _write_store(tmp_path / "store", times)
    ds = TrajectoryDataset(OmegaConf.create({"path": str(tmp_path / "store"), "horizon": 4, "init_stride": 12}))
    model = OracleModel(ds)

    item = ds[0]
    pred_states, _ = rollout_ensemble(model, item["cond"][None], item["target_clock"][None], n_members=2,
                                      norm=Normalizer(ds, "cpu"))
    torch.testing.assert_close(pred_states[0, 1], item["target_states"], rtol=1e-4, atol=1e-2)

    df = evaluate_rollout(model, ds, tmp_path / "out", n_members=2, plot_sample_grids=False,
                          n_error_dist_inits=0, num_workers=0)
    mse = df[(df.metric == "mse") & (df.member == "member_mean") & (df.variable == "all")]
    assert mse["value"].max() < 1e-6
    for name in ["R1_planetary_vorticity", "R2_geostrophic_wind", "R3_vorticity_divergence"]:
        pred = df[(df.metric == name) & (df.member == "member_mean")].sort_values(["init_time", "lead_h"])
        true = df[(df.metric == name) & (df.member == "truth")].sort_values(["init_time", "lead_h"])
        np.testing.assert_allclose(pred["value"].values, true["value"].values, rtol=1e-3)
    persistence = df[(df.metric == "mse") & (df.member == "persistence") & (df.variable == "all")]
    assert persistence["value"].min() > 0


def test_evaluate_rollout_resumes(tmp_path):
    times = _hourly("2023-01-01T00", 40)
    _write_store(tmp_path / "store", times)
    ds = TrajectoryDataset(OmegaConf.create({"path": str(tmp_path / "store"), "horizon": 2, "init_stride": 12}))
    model = OracleModel(ds)
    kwargs = dict(n_members=1, metrics=["mse"], plot_sample_grids=False, n_error_dist_inits=0, num_workers=0)

    evaluate_rollout(model, ds, tmp_path / "out", shard=(0, 2), **kwargs)
    first = load_rollout_metrics(tmp_path / "out")
    evaluate_rollout(model, ds, tmp_path / "out", shard=(0, 2), **kwargs)  # nothing left to do
    assert len(load_rollout_metrics(tmp_path / "out")) == len(first)
    evaluate_rollout(model, ds, tmp_path / "out", shard=(1, 2), **kwargs)
    assert set(load_rollout_metrics(tmp_path / "out")["init_time"]) == {ds.init_time(k) for k in range(len(ds))}
