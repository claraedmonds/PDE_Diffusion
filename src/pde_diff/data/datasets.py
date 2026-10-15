from pathlib import Path
import json
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
import einops
from omegaconf import DictConfig
from pde_diff.utils import DatasetRegistry, init_means_and_stds_era5
import xarray as xr

import pde_diff.data.const as const

@DatasetRegistry.register("dataset1")
class Dataset1(Dataset):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.transform = None
        self.dataset = None

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx) -> tuple:
        """Return a given sample from the dataset."""
        x, y = self.dataset[idx]
        return x, y

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""
        output_folder.mkdir(parents=True, exist_ok=True)
        # Implement your preprocessing logic here
        pass

@DatasetRegistry.register("fluid_data")
class FluidData(Dataset):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.data_paths = [Path(cfg.path) / 'K_data.csv', Path(cfg.path) / 'p_data.csv']
        channels = len(self.data_paths)

        for i in range(channels):
            if i == 0:
                self.data = pd.read_csv(self.data_paths[i], header=None)
            else:
                self.data = np.stack((self.data, pd.read_csv(self.data_paths[i], header=None)), axis=-1)

        dtype = torch.float64 if cfg.use_double else torch.float32
        self.data = torch.tensor(self.data, dtype=dtype)
        self.num_datapoints = len(self.data)

        assert len(self.data.shape) == 3
        self.data = generalized_b_xy_c_to_image(self.data)

    def normalize(self, arr, min_val, max_val):
        return (arr - min_val) / (max_val - min_val)

    def unnorm(self, arr, min_val, max_val):
        return arr * (max_val - min_val) + min_val

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        if index >= self.num_datapoints:
            raise IndexError('index out of range')
        return self.data[index]


def generalized_b_xy_c_to_image(tensor, pixels_x=None, pixels_y=None):
    """
    Transpose the tensor from [batch, pixel_x*pixel_y, channels, ...] to [batch, channels, ..., pixel_x, pixel_y] using einops.
    """
    if pixels_x is None or pixels_y is None:
        pixels_x = pixels_y = int(np.sqrt(tensor.shape[1]))
    num_dims = len(tensor.shape) - 2
    pattern = 'b (x y) ' + ' '.join([f'c{i}' for i in range(num_dims)]) + f' -> b ' + ' '.join([f'c{i}' for i in range(num_dims)]) + ' x y'
    return einops.rearrange(tensor, pattern, x=pixels_x, y=pixels_y)

"""
The ERA5Dataset class is responsible for loading and preprocessing the ERA5 dataset. This is a
slightly modified version of the GenCastDataset class from
 https://github.com/openclimatefix/graph_weather/blob/main/graph_weather/data/gencast_dataloader.py

It has to:
- load, normalize and concatenate (across the channel dimension) the input timesteps 0 and 1.
- load and normalize the residual between timesteps 2 and 1.

(Noise sampling and corruption of targets has been removed from original code and will be handled elsewhere)
"""
@DatasetRegistry.register("era5")
class ERA5Dataset(Dataset):
    """
    Dataset class for ERA5 training data.
    """

    def __init__(
        self,
        cfg: DictConfig,
    ):
        """
        Initialize the GenCast dataset object.

        Args the cfg should have:
            path: dataset path.
            atmospheric_features: list of features depending on pressure levels.
            single_features: list of features not depending on pressure levels.
            static_features: list of features not depending on time.
            max_year: max year to include in training set. Defaults to 2018.
            time_step: time step between predictions.
                        E.g. 12h steps correspond to time_step = 2 in a 6h dataset. Defaults to 2.
        """
        super().__init__()
        self.data = xr.open_zarr(cfg.path, chunks={})

        # Downsample longitude and latitude
        downsample_factor = cfg.get("downsample_factor", 2)  # Keep every 2nd grid point
        if downsample_factor:
            self.data = self.data.isel(
                longitude=slice(0, None, downsample_factor),
                latitude=slice(0, None, downsample_factor)
            )
            print(f"Downsampled data by a factor of {downsample_factor}.")
        self.max_year = cfg.max_year

        # Subset longitude and latitude
        lon_range = cfg.get("lon_range", None)  # Example: [0, 50]
        lat_range = cfg.get("lat_range", None)  # Example: [-50, 0]

        if lon_range:
            self.data = self.data.sel(longitude=slice(*lon_range))
            print(f"Limiting longitude to range: {lon_range}")
        if lat_range:
            self.data = self.data.sel(latitude=slice(*lat_range))
            print(f"Limiting latitude to range: {lat_range}")

        if self.data["longitude"].shape[0] % 16 != 0:
            # crop lat to multiple of 16
            print(f"Cropping longitude of size {self.data['longitude'].shape[0]} to {(self.data['longitude'].shape[0] // 16) * 16}.")
            self.data = self.data.isel(longitude=slice(0, -(self.data["longitude"].shape[0] % 16)))

        if self.data["latitude"].shape[0] % 16 != 0:
            # crop lat to multiple of 16
            print(f"Cropping latitude of size {self.data['latitude'].shape[0]} to {(self.data['latitude'].shape[0] // 16) * 16}.")
            self.data = self.data.isel(latitude=slice(0, -(self.data["latitude"].shape[0] % 16)))


        self.grid_lon = self.data["longitude"].values
        self.grid_lat = self.data["latitude"].values
        self.num_lon = len(self.grid_lon)
        self.num_lat = len(self.grid_lat)
        self.num_vars = len(self.data.keys())
        self.pressure_levels = np.array(self.data["isobaricInhPa"].values).astype(
            np.float32
        )  # Need them for loss weighting
        self.output_features_dim = len(cfg.atmospheric_features) * len(self.pressure_levels) + len(
            cfg.single_features
        )
        self.input_features_dim = self.output_features_dim + len(cfg.static_features) + 4

        self.time_step = cfg.time_step  # e.g. 2h steps correspond to time_step = 2 in a 1h dataset

        self.atmospheric_features = list(cfg.atmospheric_features)
        self.single_features = list(cfg.single_features)
        self.static_features = list(cfg.static_features)

        self.normalization_on = cfg.get("normalize", True)
        if self.normalization_on:
            self.means, self.stds, self.diff_means, self.diff_stds = init_means_and_stds_era5(self.atmospheric_features,
                                                                                              self.single_features,
                                                                                              self.static_features)

    def _normalize(self, data, means, stds):
        return (data - means[:, None, None]) / (stds[:, None, None])

    def _unnormalize(self, data, means, stds):
        return data * (stds[:, None, None]) + means[:, None, None]

    def _sin_cos_emb(self, x):
        return np.sin(2 * np.pi * x), np.cos(2 * np.pi * x)

    def _generate_clock_features(self, ds):
        # Compute sin/cos embedding for day of the year
        day_of_year = ds.time.dt.dayofyear.values
        day_of_year_grid = einops.repeat(
            day_of_year, "t -> t lon lat", lon=self.num_lon, lat=self.num_lat
        )
        sin_day_of_year, cos_day_of_year = self._sin_cos_emb(day_of_year_grid / 365.0)


        # Compute sin/cos embedding for local mean time
        hour_of_day = ds.time.dt.hour.values
        hour_of_day_grid = einops.repeat(
            hour_of_day, "t -> t lon lat", lon=self.num_lon, lat=self.num_lat
        )
        local_mean_time = hour_of_day_grid + ds["longitude"].values[None, :, None] * 4 / 60.0
        sin_local_mean_time, cos_local_mean_time = self._sin_cos_emb(local_mean_time / 24.0)

        # Stack clock features
        clock_input_data = np.stack(
            [sin_day_of_year, cos_day_of_year, sin_local_mean_time, cos_local_mean_time], axis=-1
        ).astype(np.float32)
        clock_input_data = einops.rearrange(clock_input_data, "t lon lat clock -> t clock lon lat")

        return clock_input_data

    def __len__(self):
        return sum(self.data["time.year"].values <= self.max_year) - 2 * self.time_step

    def _extract_states(self, ds) -> np.ndarray:
        """Raw (unnormalised) time-varying state channels of `ds` (which must
        keep its `time` dim), ordered atmospheric `(var lev)` then single
        features: shape (T, output_features_dim, lon, lat)."""
        atm = (
            ds[self.atmospheric_features]
            .to_array()
            .transpose("time", "longitude", "latitude", "isobaricInhPa", "variable")
            .values
        )
        states = einops.rearrange(atm, "t lon lat lev var -> t (var lev) lon lat")
        if self.single_features:
            single = (
                ds[self.single_features]
                .to_array()
                .transpose("time", "longitude", "latitude", "variable")
                .values
            )
            single = einops.rearrange(single, "t lon lat var -> t var lon lat")
            states = np.concatenate([states, single], axis=1)
        return states

    def raw_states(self, time_idx) -> np.ndarray:
        """Raw state channels at `time_idx` (list/slice of time indices) with
        the exact channel layout used for training inputs/targets."""
        return self._extract_states(self.data.isel(time=time_idx))

    def __getitem__(self, item):
        ds_conditionals = self.data.isel(time=[item, item + self.time_step])
        ds_state = self.data.isel(time=[item + 2 * self.time_step])

        # Load inputs data
        raw_states = self._extract_states(ds_conditionals)
        raw_inputs = raw_states

        if self.static_features:
            ds_conditionals_static = (
                ds_conditionals[self.static_features]
                .to_array()
                .transpose("longitude", "latitude", "variable")
                .values
            )
            ds_conditionals_static = np.stack([ds_conditionals_static] * 2, axis=0)
            ds_conditionals_static = einops.rearrange(ds_conditionals_static, "t lon lat var -> t (var) lon lat")
            raw_inputs = np.concatenate([raw_inputs, ds_conditionals_static], axis=1)


        # Normalize inputs
        if self.normalization_on:
            inputs_norm = self._normalize(raw_inputs, self.means, self.stds)
        else:
            inputs_norm = raw_inputs

        # Add time features
        clock_features = self._generate_clock_features(ds_conditionals)
        inputs = np.concatenate([inputs_norm, clock_features], axis=1)

        # Concatenate timesteps
        inputs = np.concatenate([inputs[0, :, :, :], inputs[1, :, :, :]], axis=0)
        prev_inputs = np.nan_to_num(inputs).astype(np.float32)

        # Load target data
        raw_state = self._extract_states(ds_state)[0]

        # Normalize target residuals
        raw_state_change = raw_state - raw_states[1]
        if self.normalization_on:
            state_change = self._normalize(raw_state_change, self.diff_means, self.diff_stds)
        else:
            state_change = raw_state_change
        state_change = np.nan_to_num(state_change).astype(np.float32)

        return (prev_inputs, state_change)

def increment_clock_features(clock_features: torch.Tensor, step_size: int) -> torch.Tensor:
    """
    Increment clock features by a given hourly step size using PyTorch.
    """
    # Extract the day and time and compute their angles
    sin_day_of_year, cos_day_of_year = clock_features[:, 0,:,:], clock_features[:, 1,:,:]
    sin_local_mean_time, cos_local_mean_time = clock_features[:, 2,:,:], clock_features[:, 3,:,:]
    day_of_year_angle = torch.atan2(sin_day_of_year, cos_day_of_year)
    local_mean_time_angle = torch.atan2(sin_local_mean_time, cos_local_mean_time)

    # Increment the angles
    day_of_year_angle += 2 * torch.pi * step_size / (365 * 24)  # Increment day of year angle
    local_mean_time_angle += 2 * torch.pi * step_size / 24      # Increment local mean time angle
    day_of_year_angle = day_of_year_angle % (2 * torch.pi)
    local_mean_time_angle = local_mean_time_angle % (2 * torch.pi)

    sin_day_of_year = torch.sin(day_of_year_angle)
    cos_day_of_year = torch.cos(day_of_year_angle)
    sin_local_mean_time = torch.sin(local_mean_time_angle)
    cos_local_mean_time = torch.cos(local_mean_time_angle)

    updated_clock_features = torch.stack(
        [sin_day_of_year, cos_day_of_year, sin_local_mean_time, cos_local_mean_time], dim=1
    )
    return updated_clock_features

@DatasetRegistry.register("era5_precomputed")
class PrecomputedERA5Dataset(Dataset):
    """
    Wraps pre-materialised numpy memory-mapped arrays produced by
    precompute_dataset.py.  __getitem__ is pure array indexing — no zarr /
    xarray overhead — making data loading ~100x faster than ERA5Dataset.

    The directory must contain:
        inputs.npy, targets.npy, metadata.json
    as written by precompute_dataset.py.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        out_dir = Path(cfg.path)
        with open(out_dir / "metadata.json") as f:
            meta = json.load(f)

        N = meta["N"]
        input_shape = tuple(meta["input_shape"])
        target_shape = tuple(meta["target_shape"])

        self.inputs = np.memmap(out_dir / "inputs.npy", dtype=np.float32,
                                mode="r", shape=(N, *input_shape))
        self.targets = np.memmap(out_dir / "targets.npy", dtype=np.float32,
                                 mode="r", shape=(N, *target_shape))

        # Compute start/end indices from year boundaries if min/max_year specified
        n_per_year = meta.get("n_per_year", {})
        years_sorted = sorted(n_per_year.keys())  # strings, but sort lexically (same as numeric for 4-digit years)
        cumulative = {}
        total = 0
        for y in years_sorted:
            cumulative[y] = total
            total += n_per_year[y]

        min_year = str(cfg.get("min_year", years_sorted[0] if years_sorted else None))
        max_year = str(cfg.get("max_year", years_sorted[-1] if years_sorted else None))

        if n_per_year and min_year and max_year:
            years_in_range = [y for y in years_sorted if min_year <= y <= max_year]
            if not years_in_range:
                raise ValueError(f"No years found between min_year={min_year} and max_year={max_year}. Available: {years_sorted}")
            self.start_idx = cumulative[years_in_range[0]]
            self.end_idx   = cumulative[years_in_range[-1]] + n_per_year[years_in_range[-1]]
            print(f"Using years {years_in_range[0]}–{years_in_range[-1]}: samples {self.start_idx}–{self.end_idx} ({self.end_idx - self.start_idx} total)")
        else:
            self.start_idx = 0
            self.end_idx   = N

        # Attributes expected by evaluate.py / visualize.py / VorticityLoss
        self.pressure_levels = np.array(meta["pressure_levels"], dtype=np.float32)
        self.grid_lon = np.array(meta["grid_lon"], dtype=np.float32)
        self.grid_lat = np.array(meta["grid_lat"], dtype=np.float32)
        self.means = np.array(meta["means"], dtype=np.float32)
        self.stds = np.array(meta["stds"], dtype=np.float32)
        self.diff_means = np.array(meta["diff_means"], dtype=np.float32)
        self.diff_stds = np.array(meta["diff_stds"], dtype=np.float32)

    def _normalize(self, data, means, stds):
        return (data - means[:, None, None]) / stds[:, None, None]

    def _unnormalize(self, data, means, stds):
        return data * stds[:, None, None] + means[:, None, None]

    def __len__(self) -> int:
        return self.end_idx - self.start_idx

    def __getitem__(self, idx):
        actual_idx = self.start_idx + idx
        return (
            torch.from_numpy(self.inputs[actual_idx].copy()),
            torch.from_numpy(self.targets[actual_idx].copy()),
        )


def select_init_indices(
    times: np.ndarray,
    segment_id: np.ndarray,
    horizon: int,
    stride: int = 36,
    first_hour: int = 0,
    explicit_times=None,
    time_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Indices t into an hourly trajectory that are valid rollout inits: the
    conditioning hour t-1 and every target hour up to t+horizon lie in the
    same contiguous segment as t.

    By default, inits start at the first valid `first_hour` UTC of each
    segment and are spaced `stride` hours apart (a stride that is not a
    multiple of 24 cycles through start hours). `explicit_times` (iterable of
    datetime-likes) replaces the stride selection, e.g. for extreme events;
    `time_mask` (bool per trajectory hour) further filters the result.
    """
    times = np.asarray(times).astype("datetime64[h]")
    segment_id = np.asarray(segment_id)
    T = len(times)
    t = np.arange(1, max(T - horizon, 1))
    valid = (segment_id[t - 1] == segment_id[t]) & (segment_id[t + horizon] == segment_id[t])
    valid_idx = t[valid]

    if explicit_times is not None:
        wanted = np.asarray([np.datetime64(x, "h") for x in explicit_times])
        pos = {tm: i for i, tm in enumerate(times)}
        missing = [str(w) for w in wanted if w not in pos]
        if missing:
            raise ValueError(f"init times not in trajectory: {missing}")
        idx = np.array([pos[w] for w in wanted], dtype=int)
        valid_set = set(valid_idx.tolist())
        invalid = [str(times[i]) for i in idx if i not in valid_set]
        if invalid:
            raise ValueError(f"init times without a full {horizon} h window in their segment: {invalid}")
        selected = np.unique(idx)
    else:
        hours = (times - times.astype("datetime64[D]")).astype(int)
        selected = []
        for seg in np.unique(segment_id[valid_idx]):
            seg_idx = valid_idx[segment_id[valid_idx] == seg]
            starts = seg_idx[hours[seg_idx] == first_hour]
            if len(starts) == 0:
                continue
            offset = (times[seg_idx] - times[starts[0]]).astype(int)
            selected.extend(seg_idx[(offset >= 0) & (offset % stride == 0)])
        selected = np.asarray(selected, dtype=int)

    if time_mask is not None:
        selected = selected[np.asarray(time_mask)[selected]]
    return selected


@DatasetRegistry.register("era5_trajectory")
class TrajectoryDataset(Dataset):
    """Autoregressive-rollout test set over an hourly state trajectory written
    by precompute_trajectory.py (raw states + clock features + segment ids).

    Each item is one rollout init t: the normalised conditioning
    [state_{t-1}, clock_{t-1}, state_t, clock_t] (identical layout to the
    training inputs) plus the raw true states and clock features for leads
    1..horizon.

    cfg keys: path, horizon, init_stride, init_first_hour, init_times (optional
    explicit list overriding the stride selection).
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        out_dir = Path(cfg.path)
        with open(out_dir / "metadata.json") as f:
            meta = json.load(f)

        self.states = np.load(out_dir / "states.npy", mmap_mode="r")
        self.clock = np.load(out_dir / "clock.npy", mmap_mode="r")
        self.times = np.load(out_dir / "times.npy").astype("datetime64[h]")
        self.segment_id = np.load(out_dir / "segment_id.npy")

        self.pressure_levels = np.array(meta["pressure_levels"], dtype=np.float32)
        self.atmospheric_features = list(meta["atmospheric_features"])
        self.grid_lon = np.array(meta["grid_lon"], dtype=np.float32)
        self.grid_lat = np.array(meta["grid_lat"], dtype=np.float32)
        self.means = np.array(meta["means"], dtype=np.float32)
        self.stds = np.array(meta["stds"], dtype=np.float32)
        self.diff_means = np.array(meta["diff_means"], dtype=np.float32)
        self.diff_stds = np.array(meta["diff_stds"], dtype=np.float32)

        self.horizon = int(cfg.get("horizon", 12))
        self.init_indices = select_init_indices(
            self.times,
            self.segment_id,
            horizon=self.horizon,
            stride=int(cfg.get("init_stride", 36)),
            first_hour=int(cfg.get("init_first_hour", 0)),
            explicit_times=cfg.get("init_times", None),
        )
        print(f"TrajectoryDataset: {len(self.init_indices)} inits, horizon {self.horizon} h, "
              f"{len(np.unique(self.segment_id))} segments")

    def _normalize(self, data, means, stds):
        return (data - means[:, None, None]) / stds[:, None, None]

    def _unnormalize(self, data, means, stds):
        return data * stds[:, None, None] + means[:, None, None]

    def init_time(self, idx) -> str:
        return str(self.times[self.init_indices[idx]])

    def conditioning_at(self, t: int) -> np.ndarray:
        """Normalised (2*(C+4), lon, lat) conditioning for predicting t+1."""
        blocks = []
        for i in (t - 1, t):
            state = self._normalize(np.asarray(self.states[i]), self.means, self.stds)
            blocks.extend([state, np.asarray(self.clock[i])])
        return np.nan_to_num(np.concatenate(blocks, axis=0)).astype(np.float32)

    def __len__(self) -> int:
        return len(self.init_indices)

    def __getitem__(self, idx):
        t = int(self.init_indices[idx])
        H = self.horizon
        return {
            "cond": torch.from_numpy(self.conditioning_at(t)),
            "init_state": torch.from_numpy(np.array(self.states[t], dtype=np.float32)),
            "target_states": torch.from_numpy(np.array(self.states[t + 1:t + H + 1], dtype=np.float32)),
            "target_clock": torch.from_numpy(np.array(self.clock[t + 1:t + H + 1], dtype=np.float32)),
            "init_time": self.init_time(idx),
            "index": idx,
        }


if __name__ == "__main__":
    # Test ERA5 data:
    from omegaconf import OmegaConf
    cfg = OmegaConf.load("configs/dataset/era5.yaml")

    ERA5_dataset = ERA5Dataset(cfg=cfg)
    ERA5_dataset_len = len(ERA5_dataset)
    print(f"Longitude grid size: {len(ERA5_dataset.grid_lon)}")
    print(f"Latitude grid size: {len(ERA5_dataset.grid_lat)}")
    print(f"ERA5 Dataset length: {ERA5_dataset_len}")

    sample_idx = 10
    prev_inputs, target_residuals = ERA5_dataset[sample_idx]
    print(f"Sample prev_inputs shape: {prev_inputs.shape}, Sample target_residuals shape: {target_residuals.shape}")
