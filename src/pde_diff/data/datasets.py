from pathlib import Path
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
import einops
from omegaconf import DictConfig
from pde_diff.utils import DatasetRegistry, init_means_and_stds_era5

import pde_diff.data.const as const

_SECS_PER_HOUR = 3600
_SECS_PER_DAY = 86400


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


@DatasetRegistry.register("era5")
class ERA5Dataset(Dataset):
    """
    Dataset reading from pre-processed ERA5 zarr stores (see prepare_data.py).
    Each store contains: fields (T, 15, nlon, nlat), time_unix (T,),
    longitude (nlon,), latitude (nlat,), pressure_levels (3,).
    Chunks are (1, 15, nlon, nlat) so each timestep read = one ~0.9 MB chunk.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        import zarr, glob

        paths = sorted(glob.glob(cfg.path))
        if not paths:
            raise FileNotFoundError(f"No processed zarr stores found at {cfg.path}")

        min_year = cfg.get("min_year", None)
        max_year = cfg.get("max_year", None)

        filtered = []
        for p in paths:
            try:
                year = int(Path(p).name.rsplit("_", 1)[-1])
            except ValueError:
                continue
            if min_year is not None and year < min_year:
                continue
            if max_year is not None and year > max_year:
                continue
            filtered.append(p)

        if not filtered:
            raise FileNotFoundError(
                f"No processed zarr stores in year range [{min_year}, {max_year}] at {cfg.path}"
            )

        self.time_step = cfg.time_step

        # Open fields and time_unix as explicit zarr.Array for type-safe indexed reads
        self.field_arrays = [zarr.open_array(str(Path(p) / 'fields'), mode='r') for p in filtered]
        self.time_arrays  = [zarr.open_array(str(Path(p) / 'time_unix'), mode='r') for p in filtered]

        # Build flat (store_idx, time_idx) index
        self._index: list[tuple[int, int]] = []
        for si, arr in enumerate(self.field_arrays):
            T = arr.shape[0]
            for t in range(T - 2 * self.time_step):
                self._index.append((si, t))

        # Convert coordinate arrays to numpy immediately
        first = zarr.open_group(filtered[0], mode='r')
        self.grid_lon      = np.asarray(first['longitude'], dtype=np.float32)
        self.grid_lat      = np.asarray(first['latitude'],  dtype=np.float32)
        self.pressure_levels = np.asarray(first['pressure_levels'], dtype=np.float32)
        self.num_lon = len(self.grid_lon)
        self.num_lat = len(self.grid_lat)

        self.atmospheric_features = list(cfg.atmospheric_features)
        self.single_features = list(cfg.single_features)
        self.static_features = list(cfg.static_features)
        n_atm = len(self.atmospheric_features) * len(self.pressure_levels)
        n_single = len(self.single_features)
        n_static = len(self.static_features)
        self.output_features_dim = n_atm + n_single
        self.input_features_dim = self.output_features_dim + n_static + 4  # +4 clock features

        self.normalization_on = cfg.get("normalize", True)
        if self.normalization_on:
            self.means, self.stds, self.diff_means, self.diff_stds = init_means_and_stds_era5(
                self.atmospheric_features, self.single_features, self.static_features
            )

    def __len__(self):
        return len(self._index)

    def _clock_features(self, unix_t: int) -> np.ndarray:
        """Return (4, nlon, nlat) clock features for a Unix timestamp (seconds)."""
        year = int(np.datetime64(unix_t, 's').astype('datetime64[Y]').astype(np.int64)) + 1970
        y_start = int(np.datetime64(f'{year}-01-01', 's').astype(np.int64))
        y_end   = int(np.datetime64(f'{year + 1}-01-01', 's').astype(np.int64))
        day_frac = (unix_t - y_start) / float(y_end - y_start)

        hour = (unix_t % _SECS_PER_DAY) / float(_SECS_PER_HOUR)
        local_time = ((hour + self.grid_lon * 4.0 / 60.0) % 24.0) / 24.0  # (nlon,)
        local_time_2d = local_time[:, None] + np.zeros((1, self.num_lat), dtype=np.float32)  # (nlon, nlat)

        sin_day = np.full((self.num_lon, self.num_lat), np.sin(2 * np.pi * day_frac), dtype=np.float32)
        cos_day = np.full((self.num_lon, self.num_lat), np.cos(2 * np.pi * day_frac), dtype=np.float32)
        sin_lmt = np.sin(2 * np.pi * local_time_2d).astype(np.float32)
        cos_lmt = np.cos(2 * np.pi * local_time_2d).astype(np.float32)

        return np.stack([sin_day, cos_day, sin_lmt, cos_lmt], axis=0)  # (4, nlon, nlat)

    def __getitem__(self, item):
        si, t = self._index[item]
        fields = self.field_arrays[si]
        times  = self.time_arrays[si]
        ts = self.time_step

        f0 = fields[t].astype(np.float32)           # (15, nlon, nlat)
        f1 = fields[t + ts].astype(np.float32)      # (15, nlon, nlat)
        f2 = fields[t + 2 * ts].astype(np.float32)  # (15, nlon, nlat)
        time0 = int(times[t])
        time1 = int(times[t + ts])

        if self.normalization_on:
            m, s = self.means[:, None, None], self.stds[:, None, None]
            f0n = (f0 - m) / s
            f1n = (f1 - m) / s
        else:
            f0n, f1n = f0.copy(), f1.copy()

        clk0 = self._clock_features(time0)  # (4, nlon, nlat)
        clk1 = self._clock_features(time1)  # (4, nlon, nlat)

        inp0 = np.concatenate([f0n, clk0], axis=0)   # (19, nlon, nlat)
        inp1 = np.concatenate([f1n, clk1], axis=0)   # (19, nlon, nlat)
        prev_inputs = np.concatenate([inp0, inp1], axis=0).astype(np.float32)  # (38, nlon, nlat)

        raw_residual = f2 - f1
        if self.normalization_on:
            dm, ds = self.diff_means[:, None, None], self.diff_stds[:, None, None]
            state_change = (raw_residual - dm) / ds
        else:
            state_change = raw_residual
        state_change = np.nan_to_num(state_change).astype(np.float32)

        return (prev_inputs, state_change)


def increment_clock_features(clock_features: torch.Tensor, step_size: int) -> torch.Tensor:
    """
    Increment clock features by a given hourly step size using PyTorch.
    """
    sin_day_of_year, cos_day_of_year = clock_features[:, 0,:,:], clock_features[:, 1,:,:]
    sin_local_mean_time, cos_local_mean_time = clock_features[:, 2,:,:], clock_features[:, 3,:,:]
    day_of_year_angle = torch.atan2(sin_day_of_year, cos_day_of_year)
    local_mean_time_angle = torch.atan2(sin_local_mean_time, cos_local_mean_time)

    day_of_year_angle += 2 * torch.pi * step_size / (365 * 24)
    local_mean_time_angle += 2 * torch.pi * step_size / 24
    day_of_year_angle = day_of_year_angle % (2 * torch.pi)
    local_mean_time_angle = local_mean_time_angle % (2 * torch.pi)

    sin_day_of_year = torch.sin(day_of_year_angle)
    cos_day_of_year = torch.cos(day_of_year_angle)
    sin_local_mean_time = torch.sin(local_mean_time_angle)
    cos_local_mean_time = torch.cos(local_mean_time_angle)

    return torch.stack(
        [sin_day_of_year, cos_day_of_year, sin_local_mean_time, cos_local_mean_time], dim=1
    )


@DatasetRegistry.register("era5_test")
class ERA5DatasetTest(ERA5Dataset):
    """
    Test dataset returning multi-step forecast targets.
    Returns (prev_inputs, state_change, raw_state) where the last two have a
    leading forecast_steps dimension.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.forecast_steps = cfg.get("forecast_steps", 5)
        # Rebuild index with extended lookahead
        self._index = []
        for si, arr in enumerate(self.field_arrays):
            T = arr.shape[0]
            max_t = T - 2 * self.time_step - (self.forecast_steps - 1) * self.time_step
            for t in range(max_t):
                self._index.append((si, t))

    def __getitem__(self, item):
        si, t = self._index[item]
        fields = self.field_arrays[si]
        times  = self.time_arrays[si]
        ts = self.time_step

        f0 = fields[t].astype(np.float32)
        f1 = fields[t + ts].astype(np.float32)
        time0 = int(times[t])
        time1 = int(times[t + ts])

        if self.normalization_on:
            m, s = self.means[:, None, None], self.stds[:, None, None]
            f0n = (f0 - m) / s
            f1n = (f1 - m) / s
        else:
            f0n, f1n = f0.copy(), f1.copy()

        clk0 = self._clock_features(time0)
        clk1 = self._clock_features(time1)

        inp0 = np.concatenate([f0n, clk0], axis=0)
        inp1 = np.concatenate([f1n, clk1], axis=0)
        prev_inputs = np.concatenate([inp0, inp1], axis=0).astype(np.float32)

        target_fields = np.stack(
            [fields[t + 2 * ts + step * ts].astype(np.float32)
             for step in range(self.forecast_steps)],
            axis=0,
        )  # (steps, 15, nlon, nlat)

        prev_states = np.concatenate([f1[None], target_fields[:-1]], axis=0)
        raw_residuals = target_fields - prev_states

        if self.normalization_on:
            dm, ds = self.diff_means[:, None, None], self.diff_stds[:, None, None]
            state_change = (raw_residuals - dm) / ds
        else:
            state_change = raw_residuals
        state_change = np.nan_to_num(state_change).astype(np.float32)

        return (prev_inputs, state_change, target_fields)
