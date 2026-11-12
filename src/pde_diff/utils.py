import random
import string
import numpy as np
import torch
import torch.nn.functional as F
import pde_diff.data.const as const
from types import SimpleNamespace

_ALPHABET = string.ascii_lowercase

class LayerRegistry:
    _registry = {}

    @classmethod
    def register(cls, name):
        def decorator(layer_cls):
            cls._registry[name] = layer_cls
            return layer_cls
        return decorator

    @classmethod
    def create(cls, name, *args, **kwargs):
        if name not in cls._registry:
            raise ValueError(f"Unknown layer type: {name}")
        return cls._registry[name](*args, **kwargs)

class DatasetRegistry:
    _registry = {}

    @classmethod
    def register(cls, name):
        def decorator(layer_cls):
            cls._registry[name] = layer_cls
            return layer_cls
        return decorator

    @classmethod
    def create(cls, cfg):
        if cfg.name not in cls._registry:
            raise ValueError(f"Unknown dataset: {cfg.name}")
        return cls._registry[cfg.name](cfg)

class LossRegistry:
    _registry = {}

    @classmethod
    def register(cls, name):
        def decorator(loss_cls):
            cls._registry[name] = loss_cls
            return loss_cls
        return decorator

    @classmethod
    def create(cls, cfg):
        if cfg.name not in cls._registry:
            raise ValueError(f"Unknown loss function: {cfg.name}")
        return cls._registry[cfg.name](cfg)

class SchedulerRegistry:
    _registry = {}

    @classmethod
    def register(cls, name):
        def decorator(scheduler_cls):
            cls._registry[name] = scheduler_cls
            return scheduler_cls
        return decorator

    @classmethod
    def create(cls, cfg):
        if cfg.name not in cls._registry:
            raise ValueError(f"Unknown scheduler: {cfg.name}")
        return cls._registry[cfg.name](cfg)

class ModelRegistry:
    _registry = {}

    @classmethod
    def register(cls, name):
        def decorator(model_cls):
            cls._registry[name] = model_cls
            return model_cls
        return decorator

    @classmethod
    def create(cls, cfg):
        if type(cfg) == list:
            return cls._registry[cfg[0].name](cfg)
        if cfg.name not in cls._registry:
            raise ValueError(f"Unknown model: {cfg.name}")
        return cls._registry[cfg.name](cfg)
    
class CallbackRegistry:
    _registry = {}

    @classmethod
    def register(cls, name):
        def decorator(callback_cls):
            cls._registry[name] = callback_cls
            return callback_cls
        return decorator

    @classmethod
    def create(cls, cfg):
        if cfg.name == None:
            return None
        if cfg.name not in cls._registry:
            raise ValueError(f"Unknown callback: {cfg.name}")
        return cls._registry[cfg.name](cfg)

def unique_id(existing: set[str] | None = None, length: int = 5) -> str:
    """
    Return a random a–z ID of `length` letters that isn't in `existing`.
    `existing` should be a set of already-issued IDs (optional).
    """
    existing = existing or set()
    while True:
        uid = ''.join(random.choices(_ALPHABET, k=length))
        if uid not in existing:
            return uid

def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(v) for v in d]
    else:
        return d

def init_means_and_stds_era5(atmospheric_features, single_features, static_features):
    # TODO: Handle missing features more gracefully aka actually implement them
    means = []
    stds = []
    diff_means = []
    diff_stds = []

    for var in atmospheric_features:
        try:
            means.extend(const.ERA5_MEANS[var])
            stds.extend(const.ERA5_STD[var])
            diff_means.extend(const.ERA5_DIFF_MEAN[var])
            diff_stds.extend(const.ERA5_DIFF_STD[var])
        except:
            means.extend(np.array([0.0,0.0]))
            stds.extend(np.array([1.0,1.0]))
            diff_means.extend(np.array([0.0,1.0]))
            diff_stds.extend(np.array([1.0,1.0]))

    for var in single_features:
        try:
            means.append(const.ERA5_MEANS[var])
            stds.append(const.ERA5_STD[var])
            diff_means.append(const.ERA5_DIFF_MEAN[var])
            diff_stds.append(const.ERA5_DIFF_STD[var])
        except:
            means.append(np.array([0.0,0.0]))
            stds.append(np.array([1.0,1.0]))
            diff_means.append(np.array([0.0,0.0]))
            diff_stds.append(np.array([1.0,1.0]))

    for var in static_features:
        try:
            means.append(const.ERA5_MEANS[var])
            stds.append(const.ERA5_STD[var])
        except:
            means.append(np.array([0.0,0.0]))
            stds.append(np.array([1.0,1.0]))

    return (
        np.array(means).astype(np.float32),
        np.array(stds).astype(np.float32),
        np.array(diff_means).astype(np.float32),
        np.array(diff_stds).astype(np.float32),
    )

class GradientHelper:
    """Horizontal finite differences on a regular lat/lon grid.

    Conventions:
      - Fields have layout (batch, level, lon, lat); derivatives act on the
        last two dims.
      - x points east and is periodic: longitude must cover the full circle
        with uniform spacing, so a circular central difference is exact
        across the seam.
      - y points north. Latitude may be stored north->south (as ERA5 is);
        the signed spacing dy = R * (phi[j+1] - phi[j]) then is negative,
        which makes d/dy northward automatically.
      - All spacings are ONE grid cell (dx_j = R cos(phi_j) dlambda,
        dy = R dphi). Central differences divide by 2*spacing.
      - Latitude edges (no neighbour beyond the domain) use second-order
        one-sided stencils, so derivatives are not forced to zero there.
      - The Laplacian is the local Cartesian one, d2/dx2 + d2/dy2 (no
        spherical metric terms), consistent with the f-/beta-plane QG
        equations the residuals are based on.
    """

    def __init__(self, lat_deg, lon_deg, radius):
        lat = torch.as_tensor(np.asarray(lat_deg, dtype=np.float64))
        lon = torch.as_tensor(np.asarray(lon_deg, dtype=np.float64))
        assert lat.ndim == 1 and lon.ndim == 1, "lat/lon must be 1-D coordinate arrays"
        assert lat.numel() >= 4, "need at least 4 latitude rows for the one-sided edge stencils"

        dlat = torch.diff(lat)
        assert torch.allclose(dlat, dlat[0], rtol=0, atol=1e-4), "latitude spacing must be uniform"
        dlon = torch.diff(lon)
        assert torch.allclose(dlon, dlon[0], rtol=0, atol=1e-4) and dlon[0] > 0, \
            "longitude must be increasing with uniform spacing"
        assert abs(dlon[0].item() * lon.numel() - 360.0) < 1e-3, \
            "longitude must cover the full circle (periodic x-derivative)"

        phi = torch.deg2rad(lat)
        self.nlat, self.nlon = lat.numel(), lon.numel()
        # one-cell spacings in metres; dx varies with latitude, dy is signed
        self.dx = (radius * torch.cos(phi) * torch.deg2rad(dlon[0])).view(1, 1, 1, -1)
        self.dy = radius * torch.deg2rad(dlat[0]).item()

    def _dx(self, input):
        return self.dx.to(device=input.device, dtype=input.dtype)

    def _check(self, input):
        assert input.shape[-2:] == (self.nlon, self.nlat), \
            f"expected (..., lon={self.nlon}, lat={self.nlat}), got {tuple(input.shape)}"

    def d_in_dx(self, input):
        """Eastward derivative: periodic central difference."""
        self._check(input)
        diff = torch.roll(input, shifts=-1, dims=-2) - torch.roll(input, shifts=1, dims=-2)
        return diff / (2.0 * self._dx(input))

    def d_in_dy(self, input):
        """Northward derivative: central difference inside, 2nd-order one-sided at the edges."""
        self._check(input)
        f, dy = input, self.dy
        interior = (f[..., 2:] - f[..., :-2]) / (2.0 * dy)
        first = (-3.0 * f[..., 0] + 4.0 * f[..., 1] - f[..., 2]) / (2.0 * dy)
        last = (3.0 * f[..., -1] - 4.0 * f[..., -2] + f[..., -3]) / (2.0 * dy)
        return torch.cat([first[..., None], interior, last[..., None]], dim=-1)

    def d2_in_dx2(self, input):
        """Second eastward derivative: periodic central difference."""
        self._check(input)
        diff = torch.roll(input, shifts=-1, dims=-2) - 2.0 * input + torch.roll(input, shifts=1, dims=-2)
        return diff / self._dx(input) ** 2

    def d2_in_dy2(self, input):
        """Second northward derivative: central inside, 2nd-order one-sided at the edges."""
        self._check(input)
        f, dy2 = input, self.dy ** 2
        interior = (f[..., 2:] - 2.0 * f[..., 1:-1] + f[..., :-2]) / dy2
        first = (2.0 * f[..., 0] - 5.0 * f[..., 1] + 4.0 * f[..., 2] - f[..., 3]) / dy2
        last = (2.0 * f[..., -1] - 5.0 * f[..., -2] + 4.0 * f[..., -3] - f[..., -4]) / dy2
        return torch.cat([first[..., None], interior, last[..., None]], dim=-1)

    def laplacian_horizontal(self, input):
        return self.d2_in_dx2(input) + self.d2_in_dy2(input)

    def gradient_horizontal(self, input):
        return self.d_in_dx(input), self.d_in_dy(input)

if __name__ == "__main__":
    lat = 70.0 - 0.75 * np.arange(32)
    lon = -180.0 + 0.75 * np.arange(480)
    gh = GradientHelper(lat, lon, radius=6.37e6)
    random_point = torch.randn(1, 3, 480, 32)
    gradx, grady = gh.gradient_horizontal(random_point)
    lap = gh.laplacian_horizontal(random_point)
    print(gradx.shape, grady.shape, lap.shape)
