import torch.nn as nn
import torch
import torch.nn.functional as F
import einops as ein
import math
import json
from omegaconf import ListConfig, OmegaConf
from pde_diff.utils import LossRegistry, DatasetRegistry, GradientHelper
from pde_diff.grad_utils import *
from pde_diff import data
from pde_diff.data import datasets

class PDE_loss(nn.Module):
    def __init__(self, residual_fns):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.residual_fns = residual_fns
        self.c_data = None

        if self.cfg.c_residual is not None:
            if isinstance(self.cfg.c_residual, (list, ListConfig)):
                self.c_residuals = list(self.cfg.c_residual)
            else:
                self.c_residuals = [self.cfg.c_residual for _ in residual_fns]
        else:
            self.c_residuals = [0.0 for _ in residual_fns]
        self.num_active_residuals = max(1, sum(c != 0.0 for c in self.c_residuals))

    def residual_loss(self, x0_hat, var, residual_fn):
        if self.cfg.name == 'vorticity':
            num_channels = x0_hat.shape[1]
            x0_previous = x0_hat[:, :num_channels//2, :, :]
            x0_change_pred = x0_hat[:, num_channels//2:, :, :]
            residual = residual_fn(x0_previous, x0_change_pred)
        else:
            residual = residual_fn(x0_hat)
        residual_log_likelihood = gaussian_log_likelihood(torch.zeros_like(residual), means=residual, variance=var)
        residual_loss = -1. * residual_log_likelihood
        return residual_loss.mean()

    def forward(self, model_out, target, **kwargs):
        x0_hat = kwargs.get('x0_hat', None)
        var = kwargs.get('var', None)
        total = 0.0
        for fn, w in zip(self.residual_fns, self.c_residuals):
            if w > 0.0:
                r = self.residual_loss(x0_hat, var, fn)
                total += (r * w) / self.num_active_residuals

        if self.c_data is None:
            total += self.mse(model_out, target).mean()
        else:
            total += (self.mse(model_out, target) * self.c_data[:, None, None, None]).mean()

        return total

@LossRegistry.register("mse")
class MSE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_data = None
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, model_out, target, **kwargs):
        if self.c_data is None:
            return self.mse(model_out, target).mean()
        else:
            return (self.mse(model_out, target) * self.c_data[:, None, None, None]).mean()

@LossRegistry.register("fb")
class ForecastBias(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, model_out, target, **kwargs):
        return (model_out.sum() - target.sum())/target.sum()

@LossRegistry.register("vorticity")
class VorticityLoss(PDE_loss):
    def __init__(self, cfg):
        residual_fns = [self.compute_residual_planetary_vorticity,
                        self.compute_residual_geostrophic_wind]
        self.cfg = cfg
        device_str = OmegaConf.select(cfg, "device", default=None)
        self.device = torch.device(device_str) if device_str else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.Omega = 7.292e-5 # rad s^-1
        self.T_0 = 288.15 #K
        self.R = 287 #J K^-1 kg^-1
        self.c_p = 1004 #J K^-1 kg^-1
        self.p_0 = 101325 #Pa
        self.a = 6.37e6 # m
        self.lat_range = [70, 46] # hardcoded for now
        self.lon_range = [0, 359] # hardcoded for now
        self.p = [45000, 50000, 55000] # (Pa) hardcoded for now
        self.dp = 5000 # (Pa) hardcoded for now
        self.dt = 3600 # (s) hardcoded for now

        try:
            with open("src/pde_diff/data/residual_stats.json", "r") as f:
                self.residual_stats = json.load(f)
        except FileNotFoundError:
            self.residual_stats = None

        nlat = 32  # hardcoded for now # TODO: this will vary with downsample factor
        nlon = 480  # hardcoded for now

        lats_deg = torch.linspace(self.lat_range[0],
                                  self.lat_range[1],
                                  nlat, device=self.device)
        phi = lats_deg * math.pi / 180.0
        phi0 = phi.mean()
        self.f = 2.0 * self.Omega * torch.sin(phi)[None, None, None, :]
        self.f0 = 2.0 * self.Omega * torch.sin(phi0)

        lon_grid, lat_grid, dx_grid, dy_grid = self.make_distance_grids(
            nlon=nlon,
            lon_range=self.lon_range,
            nlat=nlat,
            lat_range=self.lat_range,
            device=self.device,
            dtype=torch.float32,
        )

        self.gradient_helper = GradientHelper(grid_distances={
            'dx': dx_grid[None, None, :, :],
            'dy': dy_grid[None, None, :, :],
        })
        super().__init__(residual_fns)

    def make_distance_grids(self, nlon, lon_range, nlat, lat_range, device, dtype):
        lon = torch.linspace(lon_range[0], lon_range[1], nlon, device=device, dtype=dtype)
        lat = torch.linspace(lat_range[0], lat_range[1], nlat, device=device, dtype=dtype)
        lon_grid, lat_grid = torch.meshgrid(lon, lat, indexing='ij')  # shape: (nlon, nlat)

        lon_e_r = lon_grid.roll(1, dims=0)
        lon_e_l = lon_grid.roll(-1, dims=0)
        lat_e = lat_grid
        dx_grid = self.haversine(lon_e_l, lat_grid, lon_e_r, lat_e)

        lon_n = lon_grid
        lat_n_u = lat_grid.roll(1, dims=1)
        lat_n_d = lat_grid.roll(-1, dims=1)
        dy_grid = self.haversine(lon_grid, lat_n_u, lon_n, lat_n_d)
        return lon_grid, lat_grid, dx_grid, dy_grid

    def haversine(self, lon1, lat1, lon2, lat2):
            lon1 = torch.deg2rad(lon1)
            lat1 = torch.deg2rad(lat1)
            lon2 = torch.deg2rad(lon2)
            lat2 = torch.deg2rad(lat2)

            dlat = lat2 - lat1
            dlon = lon2 - lon1

            pi = math.pi
            dlon = (dlon + pi) % (2 * pi) - pi

            a = torch.sin(dlat / 2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2)**2
            c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a).clamp(min=1e-12))

            return self.a * c

    def set_mean_and_std(self, mean, std, diff_mean, diff_std):
        device = self.device

        self.mean = torch.as_tensor(mean).to(device)
        self.std = torch.as_tensor(std).to(device)
        self.diff_mean = torch.as_tensor(diff_mean).to(device)
        self.diff_std = torch.as_tensor(diff_std).to(device)

    def _normalize(self, x, type):
        assert self.residual_stats is not None, "Residual statistics not loaded."
        mean = self.residual_stats[type]['mean']
        std = self.residual_stats[type]['std']
        return (x - mean) / (std)

    def theta_0(self, p):
        return self.T_0 * (self.p_0 / p)**(self.R / self.c_p)

    def dx_dp(self, x):
        return ((x[:, -1] - x [:, 0]) / (self.dp * 2))

    def dxx_dpp(self, x):
        fph = x[:, 2]
        fmh = x[:, 0]
        fx = x[:, 1]
        return (fph - 2*fx + fmh) / (self.dp **2)

    def sigma(self, p):
        kappa = self.R / self.c_p
        return (self.R * self.T_0 * kappa) / (p**2)

    def get_original_states(self, x0_previous, x0_change_pred, rearrange = True):
        previous_states_unnormalized = (x0_previous * self.std[None, :, None, None]) + self.mean[None, :, None, None]
        current_state_change_unnormalized = (x0_change_pred * self.diff_std[None, :, None, None]) + self.diff_mean[None, :, None, None]
        if rearrange:
            previous_state = ein.rearrange(previous_states_unnormalized, "b (var lev) lon lat -> b lev var lon lat", lev = 3)
            current_state_change = ein.rearrange(current_state_change_unnormalized, "b (var lev) lon lat -> b lev var lon lat", lev = 3)
        else:
            previous_state = previous_states_unnormalized
            current_state_change = current_state_change_unnormalized
        return previous_state, current_state_change+previous_state

    def get_normalized_states(self, x0):
        return (x0 - self.mean[None, :, None, None])/ self.std[None, :, None, None]

    def compute_residual_geostrophic_wind(self, x0_previous, x0_change_pred, normalize=True, relative=False):
        """
        Residual of Holton eq. 6.58

        :param self: Description
        :param x0_previous: Description
        :param x0_change_pred: Description
        """
        previous, current = self.get_original_states(x0_previous, x0_change_pred)

        num_vars = current.shape[2]
        wind_u_p, wind_v_p, _, temp_p, geo_p = [previous[:, :, i] for i in range(num_vars)]
        wind_u_c, wind_v_c, _, temp_c, geo_c = [current[:, :, i] for i in range(num_vars)]

        dphi_dx_c, dphi_dy_c = self.gradient_helper.gradient_horizontal(geo_c)
        wind_geo_u_c = - dphi_dy_c / self.f0
        wind_geo_v_c = dphi_dx_c / self.f0
        residual = (wind_geo_u_c - wind_u_c).abs() + (wind_geo_v_c - wind_v_c).abs()

        if relative:
            wind_ageo_u_c = wind_geo_u_c - wind_u_c
            wind_ageo_v_c = wind_geo_v_c - wind_v_c
            return (wind_ageo_u_c.abs())/(wind_ageo_u_c.abs()+wind_geo_u_c.abs()+1e-10), (wind_ageo_v_c.abs())/(wind_ageo_v_c.abs()+wind_geo_v_c.abs()+1e-10)
        if normalize:
            residual = self._normalize(residual, 'geostrophic_wind')
        return residual

    def compute_residual_planetary_vorticity(self, x0_previous, x0_change_pred, normalize=True):
        """
        Residual of Holton eq. 6.65

        :param self: Description
        :param x0_previous: Description
        :param x0_change_pred: Description
        """
        previous, current = self.get_original_states(x0_previous, x0_change_pred)

        num_vars = current.shape[2]
        wind_u_p, wind_v_p, _, temp_p, geo_p = [previous[:, :, i] for i in range(num_vars)]
        wind_u_c, wind_v_c, _, temp_c, geo_c = [current[:, :, i] for i in range(num_vars)]

        q_p, q_c = self.get_q(x0_previous, x0_change_pred)

        dphi_dx_c, dphi_dy_c = self.gradient_helper.gradient_horizontal(geo_c)
        wind_geo_u_c = -dphi_dy_c / self.f0
        wind_geo_v_c = dphi_dx_c / self.f0

        dq_dx_c, dq_dy_c = self.gradient_helper.gradient_horizontal(q_c[:, None])
        wind_nabla_dot = wind_geo_u_c[:, 1] * dq_dx_c[:,0] + wind_geo_v_c[:, 1] * dq_dy_c[:,0]
        dq_dt = (q_c - q_p) / (self.dt)
        residual = dq_dt + wind_nabla_dot
        if normalize:
            residual = self._normalize(residual, 'planetary_vorticity')
        return residual

    def get_q(self, x0_previous, x0_change_pred):
        previous, current = self.get_original_states(x0_previous, x0_change_pred)
        num_vars = current.shape[2]
        wind_u_p, wind_v_p, _, temp_p, geo_p = [previous[:, :, i] for i in range(num_vars)]
        wind_u_c, wind_v_c, _, temp_c, geo_c = [current[:, :, i] for i in range(num_vars)]
        device = x0_change_pred.device
        dtype = x0_change_pred.dtype
        p = torch.tensor(self.p, device=device, dtype=dtype)[None, :, None, None]
        sigma = self.sigma(p)
        lap_geo = self.gradient_helper.laplacian_horizontal(geo_c)
        A = self.f0 / sigma
        term_vert = (2 / p[:, 1]) * A[:, 1] * self.dx_dp(geo_c)  + A[:, 1] * self.dxx_dpp(geo_c)
        q_c = (1/self.f0 * lap_geo + self.f)[:,1] + term_vert
        lap_geo_p = self.gradient_helper.laplacian_horizontal(geo_p)
        term_vert_p = (2 / p[:, 1]) * A[:, 1] * self.dx_dp(geo_p)  + A[:, 1] * self.dxx_dpp(geo_p)
        q_p = (1/self.f0 * lap_geo_p + self.f)[:,1] + term_vert_p
        return q_p, q_c

    def vorticity_residual_loss(self, x0_hat, var):
        num_channels = x0_hat.shape[1]
        num_batch = x0_hat.shape[0]
        x0_previous = x0_hat[:, :num_channels//2, :, :]
        x0_change_pred = x0_hat[:, num_channels//2:, :, :]

        residual = self.compute_residual(x0_previous, x0_change_pred)
        residual_log_likelihood = gaussian_log_likelihood(torch.zeros_like(residual), means=residual, variance=var[:num_batch])
        residual_loss = -1. * residual_log_likelihood
        return residual_loss.mean()


def gaussian_log_likelihood(x, means, variance, return_full = False):
    centered_x = x - means
    expand_dims = centered_x.dim() - variance.dim()
    variance_broadcast = variance.view(*variance.shape, *([1] * expand_dims))
    squared_diffs = (centered_x ** 2) / variance_broadcast
    if return_full:
        log_likelihood = -0.5 * (squared_diffs + torch.log(variance) + torch.log(2 * torch.pi)) # full log likelihood with constant terms
    else:
        log_likelihood = -0.5 * squared_diffs

    # avoid log(0)
    log_likelihood = torch.clamp(log_likelihood, min=-27.6310211159)

    return log_likelihood

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from omegaconf import OmegaConf

    cfg = OmegaConf.load("./configs/dataset/era5.yaml")
    cfg_loss = OmegaConf.create(
        {
            "name": "vorticity",
            "c_residual": 1.0,
            "device": "cpu",
        }
    )

    dataset = DatasetRegistry.create(cfg)
    cfg.normalize = False
    dataset_no_norm = DatasetRegistry.create(cfg)

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    loader_no_norm = DataLoader(
        dataset_no_norm,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    loss = LossRegistry.create(cfg_loss)
    loss.set_mean_and_std(dataset.means, dataset.stds, dataset.diff_means, dataset.diff_stds)
    loss.device = torch.device("cpu")
    num_vars = 4

    it = iter(loader)
    first_batch = next(it)
    batch = first_batch
    x, y = batch
    num_vars = (x.shape[1] - 8) // 6
    previous = x[:, num_vars*3+4:-4]
    current = y

    it_no_norm = iter(loader_no_norm)
    first_batch_no_norm = next(it_no_norm)
    batch_no_norm = first_batch_no_norm
    x_no_norm, y_no_norm = batch_no_norm
    num_vars = (x_no_norm.shape[1] - 8) // 6
    previous_no_norm = ein.rearrange(x_no_norm[:, num_vars*3+4:-4], "b (var lev) lon lat -> b lev var lon lat", lev = 3)
    current_no_norm = ein.rearrange(y_no_norm, "b (var lev) lon lat -> b lev var lon lat", lev = 3)

    prev_no_normed, current_no_normed = loss.get_original_states(previous, current)
    change_no_normed = current_no_normed - prev_no_normed

    diff_prev = (prev_no_normed-previous_no_norm).abs() # abs diff between denormed and original
    diff_change = (change_no_normed - current_no_norm).abs() # abs diff between denormed change and original change

    for i in range(num_vars):
        print(f"Max abs diff variable {i}: prev {diff_prev[:, :, i].max()}, change {diff_change[:, :, i].max()}")
        print(f"Relative max abs diff variable {i}: prev {diff_prev[:, :, i].max() / (previous_no_norm[:, :, i].abs().max()+1e-12)}, change {diff_change[:, :, i].max()/(current_no_norm[:, :, i].abs().max()+1e-12)}")
        print(f"2 norm diff variable {i}: prev {torch.norm(diff_prev[:, :, i])}, change {torch.norm(diff_change[:, :, i])}")

    r_era5_pv = loss.compute_residual_planetary_vorticity(previous, current, normalize=False).abs().mean()
    r_era5_geo_wind = loss.compute_residual_geostrophic_wind(previous, current, normalize=False).abs().mean()

    random_prev = torch.randn_like(previous)
    random_curr = torch.randn_like(current)
    r_rand_pv = loss.compute_residual_planetary_vorticity(random_prev, random_curr, normalize=False).abs().mean()
    r_rand_geo_wind = loss.compute_residual_geostrophic_wind(random_prev, random_curr, normalize=False).abs().mean()

    print("_______________")
    print("ERA5 residual planetary:", r_era5_pv.item())
    print("Random residual planetary:", r_rand_pv.item())
    print("_______________")
    print("ERA5 residual geo wind:", r_era5_geo_wind.item())
    print("Random residual geo wind:", r_rand_geo_wind.item())

    # ERA5 (normalized)
    r_era5_pv_norm = (loss.compute_residual_planetary_vorticity(previous, current, normalize=True).abs().mean())
    r_era5_geo_wind = (loss.compute_residual_geostrophic_wind(previous, current, normalize=True).abs().mean())

    random_prev = torch.randn_like(previous)
    random_curr = torch.randn_like(current)

    r_rand_pv_norm = (loss.compute_residual_planetary_vorticity(random_prev, random_curr, normalize=True).abs().mean())
    r_rand_geo_wind = (loss.compute_residual_geostrophic_wind(random_prev, random_curr, normalize=True).abs().mean())

    print("_______________")
    print("ERA5 residual planetary (normalized):", r_era5_pv_norm.item())
    print("Random residual planetary (normalized):", r_rand_pv_norm.item())
    print("_______________")
    print("_______________")
    print("ERA5 residual geo wind:", r_era5_geo_wind.item())
    print("Random residual geo wind:", r_rand_geo_wind.item())
