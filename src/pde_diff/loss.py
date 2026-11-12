import torch.nn as nn
import torch
import torch.nn.functional as F
import einops as ein
import math
import numpy as np
import json
from omegaconf import ListConfig, OmegaConf
from pde_diff.utils import LossRegistry, DatasetRegistry, GradientHelper
from pde_diff.grad_utils import *
from pde_diff import data
from pde_diff.data import datasets

# Floor applied to the per-element Gaussian log-likelihood (≈ ln(1e-12)).
# Elements at the floor contribute a constant loss and zero gradient.
# Only used by the legacy "bastek_clamped" weighting (copied from Bastek et
# al.'s toy-problem code); see Bugs_in_gradient.md, section 9.
LOG_LIKELIHOOD_FLOOR = -27.6310211159


# ---------------------------------------------------------------------------
# Physics-residual weightings
#
# Each weighting maps a normalised residual r (any shape, batch first) and a
# context dict of per-sample diffusion quantities to a per-element loss, plus
# optional diagnostics. The overall weight c_residual is applied outside.
#
# Context keys (all per sample, shape (B,) unless noted):
#   "var_t"         posterior variance Sigma_t of the sampled timestep
#   "snr_weight"    min(SNR_t, 5), the p2 weight of the data term (or None)
#   "intrinsic_var" scalar: ERA5's own E[r^2] for this residual
#
# To add a weighting, write a function with this signature and register it in
# RESIDUAL_WEIGHTINGS, e.g. an SNR-weighted squared residual:
#     def _weight_snr(r, ctx): return _per_sample(ctx["snr_weight"], r) * r**2, {}
# ---------------------------------------------------------------------------

def _per_sample(v, ref):
    """Reshape a per-sample (B,) tensor to broadcast against `ref` (B, ...)."""
    return v.view(-1, *([1] * (ref.dim() - 1)))


def _weight_intrinsic(r, ctx):
    """Virtual Gaussian likelihood whose variance is the diffusion uncertainty
    PLUS the intrinsic residual variance of the real atmosphere:
        0.5 * r^2 / (Sigma_t + sigma^2_ERA5).
    The physics is only approximate (ERA5 itself has r != 0), so the residual
    is not required to vanish; the loss stays bounded at small t."""
    return 0.5 * r**2 / (_per_sample(ctx["var_t"], r) + ctx["intrinsic_var"]), {}


def _weight_bastek(r, ctx):
    """Bastek et al. (ICLR 2025) virtual likelihood without the clamp:
    0.5 * r^2 / Sigma_t (assumes the physics holds exactly)."""
    return 0.5 * r**2 / _per_sample(ctx["var_t"], r), {}


def _weight_bastek_clamped(r, ctx):
    """Legacy: 0.5 * r^2 / Sigma_t capped at -LOG_LIKELIHOOD_FLOOR (zero
    gradient above the cap). Kept only to reproduce earlier runs."""
    log_likelihood = torch.clamp(-0.5 * r**2 / _per_sample(ctx["var_t"], r), min=LOG_LIKELIHOOD_FLOOR)
    clamp_frac = (log_likelihood.detach() <= LOG_LIKELIHOOD_FLOOR).float().mean()
    return -log_likelihood, {"clamp_frac": clamp_frac}


RESIDUAL_WEIGHTINGS = {
    "intrinsic": _weight_intrinsic,
    "bastek": _weight_bastek,
    "bastek_clamped": _weight_bastek_clamped,
}


class PDE_loss(nn.Module):
    def __init__(self, residual_fns, residual_names=None):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.residual_fns = residual_fns
        self.residual_names = list(residual_names) if residual_names is not None else [f"r{i}" for i in range(len(residual_fns))]
        self.c_data = None
        # Detached per-component diagnostics from the most recent forward(),
        # read and logged by DiffusionModel (keys without split prefix).
        self.last_components = {}
        self.log_grad_norms = OmegaConf.select(self.cfg, "log_grad_norms", default=True)
        # How the physics residuals enter the loss (see RESIDUAL_WEIGHTINGS).
        self.residual_weighting = OmegaConf.select(self.cfg, "residual_weighting", default="intrinsic")
        assert self.residual_weighting in RESIDUAL_WEIGHTINGS, \
            f"loss.residual_weighting must be one of {list(RESIDUAL_WEIGHTINGS)}, got {self.residual_weighting!r}"

        if self.cfg.c_residual is not None:
            if isinstance(self.cfg.c_residual, (list, ListConfig)):
                self.c_residuals = list(self.cfg.c_residual)
            else:
                self.c_residuals = [self.cfg.c_residual for _ in residual_fns]
        else:
            self.c_residuals = [0.0 for _ in residual_fns]
        self.num_active_residuals = max(1, sum(c != 0.0 for c in self.c_residuals))

    def intrinsic_residual_variance(self, name):
        """E[r^2] of the normalised residual `name` on real data (subclasses
        override); used as the intrinsic variance of the "intrinsic" weighting."""
        return 1.0

    def residual_loss(self, x0_hat, var, residual_fn, name):
        """Mean physics loss of one residual and its diagnostics (dict)."""
        if self.cfg.name == 'vorticity':
            num_channels = x0_hat.shape[1]
            x0_previous = x0_hat[:, :num_channels//2, :, :]
            x0_change_pred = x0_hat[:, num_channels//2:, :, :]
            residual = residual_fn(x0_previous, x0_change_pred)
        else:
            residual = residual_fn(x0_hat)
        # intrinsic variance = ERA5's own E[r^2] for this residual (residual_stats.json)
        ctx = {"var_t": var, "snr_weight": self.c_data, "intrinsic_var": float(self.intrinsic_residual_variance(name))}
        per_element, diagnostics = RESIDUAL_WEIGHTINGS[self.residual_weighting](residual, ctx)
        return per_element.mean(), diagnostics

    def forward(self, model_out, target, **kwargs):
        x0_hat = kwargs.get('x0_hat', None)
        var = kwargs.get('var', None)

        if self.c_data is None:
            data_loss = self.mse(model_out, target).mean()
        else:
            data_loss = (self.mse(model_out, target) * self.c_data[:, None, None, None]).mean()

        phys_terms = {}
        components = {"data_loss": data_loss.detach()}
        for name, fn, w in zip(self.residual_names, self.residual_fns, self.c_residuals):
            if w > 0.0:
                r, diagnostics = self.residual_loss(x0_hat, var, fn, name)
                phys_terms[name] = (r * w) / self.num_active_residuals
                components[f"phys_loss_{name}"] = phys_terms[name].detach()
                if "clamp_frac" in diagnostics:
                    components[f"phys_clamp_frac_{name}"] = diagnostics["clamp_frac"]

        total = data_loss + sum(phys_terms.values())
        if phys_terms:
            components["phys_loss_total"] = (total - data_loss).detach()

        # Gradient norm of each term w.r.t. the network output. Only backprops
        # through the loss graph (not the network), so it is cheap. Skipped in
        # val/test where autograd is disabled.
        if self.log_grad_norms and torch.is_grad_enabled() and model_out.requires_grad:
            for name, term in [("data", data_loss), *((f"phys_{n}", t) for n, t in phys_terms.items())]:
                (g,) = torch.autograd.grad(term, model_out, retain_graph=True, allow_unused=True)
                components[f"gradnorm_{name}"] = g.norm().detach() if g is not None else torch.zeros((), device=model_out.device)

        self.last_components = components
        return total

@LossRegistry.register("mse")
class MSE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_data = None
        self.mse = nn.MSELoss(reduction='none')
        self.last_components = {}

    def forward(self, model_out, target, **kwargs):
        if self.c_data is None:
            loss = self.mse(model_out, target).mean()
        else:
            loss = (self.mse(model_out, target) * self.c_data[:, None, None, None]).mean()
        self.last_components = {"data_loss": loss.detach()}
        return loss

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
                        self.compute_residual_geostrophic_wind,
                        self.compute_residual_vorticity_divergence]
        self.cfg = cfg
        device_str = OmegaConf.select(cfg, "device", default=None)
        self.device = torch.device(device_str) if device_str else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.Omega = 7.292e-5 # rad s^-1
        self.T_0 = 288.15 #K
        self.R = 287 #J K^-1 kg^-1
        self.c_p = 1004 #J K^-1 kg^-1
        self.p_0 = 101325 #Pa
        self.a = 6.37e6 # m
        self.p = [45000, 50000, 55000] # (Pa) hardcoded for now
        self.dp = 5000 # (Pa) hardcoded for now
        self.dt = 3600 # (s) hardcoded for now

        try:
            with open("src/pde_diff/data/residual_stats.json", "r") as f:
                self.residual_stats = json.load(f)
        except FileNotFoundError:
            self.residual_stats = None

        # Default grid = the downsampled ERA5 grid used for training
        # (0.25 deg native, downsample_factor=3, latitude 70N -> 46.75N stored
        # north->south and cropped to 32 rows, full longitude circle).
        # Callers with a dataset at hand should call set_grid(dataset.grid_lat,
        # dataset.grid_lon) so the coordinates come from the data itself.
        self.set_grid(70.0 - 0.75 * np.arange(32), -180.0 + 0.75 * np.arange(480))
        super().__init__(residual_fns, residual_names=["pv", "geo", "vd"])

    def set_grid(self, grid_lat, grid_lon):
        """(Re)build the Coriolis parameters and the finite-difference helper
        from 1-D latitude/longitude coordinates in degrees."""
        grid_lat = np.asarray(grid_lat, dtype=np.float64)
        grid_lon = np.asarray(grid_lon, dtype=np.float64)
        if hasattr(self, "gradient_helper"):
            assert (len(grid_lat), len(grid_lon)) == (self.gradient_helper.nlat, self.gradient_helper.nlon), \
                "set_grid: grid size differs from the model's spatial dims"
        self.grid_lat, self.grid_lon = grid_lat, grid_lon
        phi = torch.deg2rad(torch.as_tensor(grid_lat, dtype=torch.float32, device=self.device))
        self.f = 2.0 * self.Omega * torch.sin(phi)[None, None, None, :]
        self.f0 = 2.0 * self.Omega * torch.sin(phi.mean())
        self.gradient_helper = GradientHelper(grid_lat, grid_lon, radius=self.a)

    def set_mean_and_std(self, mean, std, diff_mean, diff_std):
        device = self.device

        self.mean = torch.as_tensor(mean).to(device)
        self.std = torch.as_tensor(std).to(device)
        self.diff_mean = torch.as_tensor(diff_mean).to(device)
        self.diff_std = torch.as_tensor(diff_std).to(device)

    # residual name -> (key in residual_stats.json, standardised with mean subtraction?)
    RESIDUAL_STATS_KEYS = {
        "pv": ("planetary_vorticity", True),
        "geo": ("geostrophic_wind", False),  # signed (du, dv): divided by the RMS, target stays 0
        "vd": ("vorticity_divergence", True),
    }
    # Latitude rows dropped at each edge of every residual. There d/dy needs a
    # one-sided stencil, which on ERA5 inflates the R3 residual ~2.5x (B 0.30
    # vs 0.19 in the interior) and amplifies grid-scale noise; see
    # Bugs_in_gradient.md, section 12. Derivatives are still computed on the
    # full grid, only the residuals (loss, stats, balance, diagnostics) are cropped.
    EDGE_ROWS = 1

    def interior(self, x):
        """Drop EDGE_ROWS latitude rows (last axis) at each edge."""
        k = self.EDGE_ROWS
        return x[..., k:x.shape[-1] - k] if k else x

    def pad_edge_rows(self, x, value=float("nan")):
        """Inverse of `interior` for plotting: pad the dropped rows with `value`."""
        k = self.EDGE_ROWS
        return F.pad(x, (k, k), value=value) if k else x

    def intrinsic_residual_variance(self, name):
        """E[r^2] of the normalised residual on ERA5. Every residual is
        normalised so that this is 1: R1 and R3 are standardised (mean 0,
        std 1), and R2 is divided by its RMS (E[r^2] = 1 by construction).
        Kept as a method so a residual with a different normalisation can
        return its own value."""
        return 1.0

    def _normalize(self, x, type, center=True):
        """Scale a raw residual by its ERA5 statistics (residual_stats.json).
        center=True standardises, (x - mean) / std: right for signed residuals
        whose physical target 0 is ~the ERA5 mean (R1, R3). center=False
        divides by the RMS = sqrt(mean^2 + std^2) without subtracting the
        mean, so the target stays exact balance (0) rather than the
        ERA5-average imbalance and E[r^2] on ERA5 is 1 (R2)."""
        assert self.residual_stats is not None, "Residual statistics not loaded."
        stats = self.residual_stats[type]
        if not center:
            return x / math.sqrt(stats['mean'] ** 2 + stats['std'] ** 2)
        return (x - stats['mean']) / stats['std']

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

    def compute_residual_geostrophic_wind(self, x0_previous, x0_change_pred, normalize=True, relative=False,
                                          return_terms=False):
        """
        Residual of Holton eq. 6.58: the signed ageostrophic wind
        R2 = (u_g - u, v_g - v), shaped (b, 2, lev, lon, lat) (m/s before
        normalisation), with the QG geostrophic wind V_g = f0^-1 k x grad(Phi).
        Squared in the loss this gives |V_g - V|^2 = du^2 + dv^2: smooth,
        independent of the grid orientation, and linear in the state before
        squaring. Both components share one ERA5 RMS, so the loss stays
        rotation-invariant after normalisation.

        relative: False -> R2. True / "components" -> (rel_u, rel_v) with
            rel_u = |u_g - u| / (|u_g - u| + |u_g|) (same for v), each in [0, 1].
            "vector" -> |V_g - V| / (|V_g - V| + |V_g|), in [0, 1], using vector
            magnitudes (robust where a single component of V_g crosses zero).
        return_terms: return the term groups instead (see `balance_ratio`):
            [[u_g, -u], [v_g, -v]], i.e. one group per component of R2.
        """
        previous, current = self.get_original_states(x0_previous, x0_change_pred)

        num_vars = current.shape[2]
        wind_u_p, wind_v_p, _, temp_p, geo_p = [previous[:, :, i] for i in range(num_vars)]
        wind_u_c, wind_v_c, _, temp_c, geo_c = [current[:, :, i] for i in range(num_vars)]

        wind_geo_u_c, wind_geo_v_c = self.geostrophic_wind(geo_c)
        wind_geo_u_c, wind_geo_v_c, wind_u_c, wind_v_c = map(self.interior, (wind_geo_u_c, wind_geo_v_c, wind_u_c, wind_v_c))
        if return_terms:
            return [[wind_geo_u_c, -wind_u_c], [wind_geo_v_c, -wind_v_c]]
        wind_ageo_u_c = wind_geo_u_c - wind_u_c
        wind_ageo_v_c = wind_geo_v_c - wind_v_c
        residual = torch.stack([wind_ageo_u_c, wind_ageo_v_c], dim=1)
        if relative is True or relative == "components":
            return (wind_ageo_u_c.abs())/(wind_ageo_u_c.abs()+wind_geo_u_c.abs()+1e-10), (wind_ageo_v_c.abs())/(wind_ageo_v_c.abs()+wind_geo_v_c.abs()+1e-10)
        if relative == "vector":
            ageo = torch.sqrt(wind_ageo_u_c**2 + wind_ageo_v_c**2)
            geo = torch.sqrt(wind_geo_u_c**2 + wind_geo_v_c**2)
            return ageo / (ageo + geo + 1e-10)
        assert relative is False, f"unknown relative={relative!r}"
        if normalize:
            # Target 0: scale by the RMS only, do not subtract the mean.
            residual = self._normalize(residual, 'geostrophic_wind', center=False)
        return residual

    def geostrophic_wind(self, geo):
        """QG geostrophic wind V_g = f0^-1 k x grad(Phi) of geopotential `geo`
        (b, lev, lon, lat) in m^2/s^2; returns (u_g, v_g) in m/s.

        f0 on purpose: R2 enforces V = V_g with the QG definition of the geostrophic
        wind, the same V_g that advects q in R1. Over 46-70N f varies by ~+-13%, which
        gives ERA5 a latitude-dependent ageostrophic bias of ~+-1 m/s under this
        definition (known QG limitation)."""
        dphi_dx, dphi_dy = self.gradient_helper.gradient_horizontal(geo)
        return -dphi_dy / self.f0, dphi_dx / self.f0

    def _qg_advection(self, geo, q):
        """Geostrophic advection of q at the middle level, split into its x and y
        parts: (u_g dq/dx, v_g dq/dy), with V_g = f0^-1 k x grad(Phi)."""
        wind_geo_u, wind_geo_v = self.geostrophic_wind(geo)
        dq_dx, dq_dy = self.gradient_helper.gradient_horizontal(q[:, None])
        return wind_geo_u[:, 1] * dq_dx[:, 0], wind_geo_v[:, 1] * dq_dy[:, 0]

    def compute_residual_planetary_vorticity(self, x0_previous, x0_change_pred, normalize=True, return_terms=False):
        """
        Residual of Holton eq. 6.65 (QG PV conservation), time-centred:

            R1 = (q_c - q_p) / dt + 1/2 [V_g . grad q]_p + 1/2 [V_g . grad q]_c

        The 1 h difference (q_c - q_p)/dt is centred at t - dt/2, so the
        advection term is averaged over the previous (p) and current (c) state
        (trapezoidal rule) to be evaluated at the same time, making R1
        second-order accurate in time.
        """
        previous, current = self.get_original_states(x0_previous, x0_change_pred)

        num_vars = current.shape[2]
        wind_u_p, wind_v_p, _, temp_p, geo_p = [previous[:, :, i] for i in range(num_vars)]
        wind_u_c, wind_v_c, _, temp_c, geo_c = [current[:, :, i] for i in range(num_vars)]

        q_p, q_c = self.get_q(x0_previous, x0_change_pred)

        adv_x_p, adv_y_p = self._qg_advection(geo_p, q_p)
        adv_x_c, adv_y_c = self._qg_advection(geo_c, q_c)
        adv_x = 0.5 * (adv_x_p + adv_x_c)  # u_g dq/dx at t - dt/2
        adv_y = 0.5 * (adv_y_p + adv_y_c)  # v_g dq/dy at t - dt/2
        dq_dt = (q_c - q_p) / (self.dt)
        dq_dt, adv_x, adv_y = map(self.interior, (dq_dt, adv_x, adv_y))
        if return_terms:  # R1 = dq/dt + u_g dq/dx + v_g dq/dy (time-centred)
            return [[dq_dt, adv_x, adv_y]]
        residual = dq_dt + adv_x + adv_y
        if normalize:
            residual = self._normalize(residual, 'planetary_vorticity')
        return residual

    def compute_residual_vorticity_divergence(self, x0_previous, x0_change_pred, normalize=True, return_terms=False):
        """
        Residual of the barotropic vorticity-divergence equation (see Equations.md),
        time-centred like R1:

            R3 = (zeta_c - zeta_p) / dt + 1/2 [N]_p + 1/2 [N]_c,
            N  = u dzeta/dx + v dzeta/dy + (zeta + f) delta

        (zeta_c - zeta_p)/dt is centred at t - dt/2, so the advection and
        divergence terms N are averaged over the previous and current state.
        """
        previous, current = self.get_original_states(x0_previous, x0_change_pred)

        num_vars = current.shape[2]
        wind_u_p, wind_v_p, _, temp_p, geo_p = [previous[:, :, i] for i in range(num_vars)]
        wind_u_c, wind_v_c, _, temp_c, geo_c = [current[:, :, i] for i in range(num_vars)]

        du_dx_c, du_dy_c = self.gradient_helper.gradient_horizontal(wind_u_c)
        dv_dx_c, dv_dy_c = self.gradient_helper.gradient_horizontal(wind_v_c)
        zeta_c = dv_dx_c - du_dy_c
        delta_c = du_dx_c + dv_dy_c

        du_dx_p, du_dy_p = self.gradient_helper.gradient_horizontal(wind_u_p)
        dv_dx_p, dv_dy_p = self.gradient_helper.gradient_horizontal(wind_v_p)
        zeta_p = dv_dx_p - du_dy_p
        delta_p = du_dx_p + dv_dy_p

        dzeta_dt = (zeta_c - zeta_p) / self.dt

        dzeta_dx_c, dzeta_dy_c = self.gradient_helper.gradient_horizontal(zeta_c)
        dzeta_dx_p, dzeta_dy_p = self.gradient_helper.gradient_horizontal(zeta_p)

        # trapezoidal average over previous and current state (centred at t - dt/2)
        adv_x = 0.5 * (wind_u_p * dzeta_dx_p + wind_u_c * dzeta_dx_c)
        adv_y = 0.5 * (wind_v_p * dzeta_dy_p + wind_v_c * dzeta_dy_c)
        div_term = 0.5 * ((zeta_p + self.f) * delta_p + (zeta_c + self.f) * delta_c)
        dzeta_dt, adv_x, adv_y, div_term = map(self.interior, (dzeta_dt, adv_x, adv_y, div_term))

        if return_terms:  # R3 = dzeta/dt + u dzeta/dx + v dzeta/dy + (zeta + f) delta (time-centred)
            return [[dzeta_dt, adv_x, adv_y, div_term]]
        residual = dzeta_dt + adv_x + adv_y + div_term
        if normalize:
            residual = self._normalize(residual, 'vorticity_divergence')
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


def balance_ratio(term_groups):
    """Sums needed for the balance index B = mean|R| / mean(sum_k |term_k|).

    `term_groups` is what compute_residual_*(..., return_terms=True) returns:
    a list of groups of signed terms with R = sum_groups |sum_k term_k|.
    By the triangle inequality 0 <= B <= 1: B ~ 0 means the terms cancel
    (equation well satisfied), B ~ 1 means they don't cancel at all (no
    balance). Returns (sum of |R|, sum of sum_k |term_k|, count) so the index
    can be accumulated over many samples: B = total_num / total_den.
    """
    num = sum(sum(group).abs() for group in term_groups)
    den = sum(sum(t.abs() for t in group) for group in term_groups)
    return num.sum().item(), den.sum().item(), num.numel()


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
    log_likelihood = torch.clamp(log_likelihood, min=LOG_LIKELIHOOD_FLOOR)

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
    loss.set_grid(dataset.grid_lat, dataset.grid_lon)
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
    r_era5_vd = loss.compute_residual_vorticity_divergence(previous, current, normalize=False).abs().mean()

    random_prev = torch.randn_like(previous)
    random_curr = torch.randn_like(current)
    r_rand_pv = loss.compute_residual_planetary_vorticity(random_prev, random_curr, normalize=False).abs().mean()
    r_rand_geo_wind = loss.compute_residual_geostrophic_wind(random_prev, random_curr, normalize=False).abs().mean()
    r_rand_vd = loss.compute_residual_vorticity_divergence(random_prev, random_curr, normalize=False).abs().mean()

    print("_______________")
    print("ERA5 residual planetary:", r_era5_pv.item())
    print("Random residual planetary:", r_rand_pv.item())
    print("_______________")
    print("ERA5 residual geo wind:", r_era5_geo_wind.item())
    print("Random residual geo wind:", r_rand_geo_wind.item())
    print("_______________")
    print("ERA5 residual vorticity-divergence:", r_era5_vd.item())
    print("Random residual vorticity-divergence:", r_rand_vd.item())

    # ERA5 (normalized)
    r_era5_pv_norm = (loss.compute_residual_planetary_vorticity(previous, current, normalize=True).abs().mean())
    r_era5_geo_wind = (loss.compute_residual_geostrophic_wind(previous, current, normalize=True).abs().mean())
    r_era5_vd_norm = (loss.compute_residual_vorticity_divergence(previous, current, normalize=True).abs().mean())

    random_prev = torch.randn_like(previous)
    random_curr = torch.randn_like(current)

    r_rand_pv_norm = (loss.compute_residual_planetary_vorticity(random_prev, random_curr, normalize=True).abs().mean())
    r_rand_geo_wind = (loss.compute_residual_geostrophic_wind(random_prev, random_curr, normalize=True).abs().mean())
    r_rand_vd_norm = (loss.compute_residual_vorticity_divergence(random_prev, random_curr, normalize=True).abs().mean())

    print("_______________")
    print("ERA5 residual planetary (normalized):", r_era5_pv_norm.item())
    print("Random residual planetary (normalized):", r_rand_pv_norm.item())
    print("_______________")
    print("_______________")
    print("ERA5 residual geo wind:", r_era5_geo_wind.item())
    print("Random residual geo wind:", r_rand_geo_wind.item())
    print("_______________")
    print("ERA5 residual vorticity-divergence (normalized):", r_era5_vd_norm.item())
    print("Random residual vorticity-divergence (normalized):", r_rand_vd_norm.item())
