"""Analytic tests for the horizontal finite differences (GradientHelper) and
for the physics residuals that use them (VorticityLoss).

Grid: the default VorticityLoss grid, i.e. the downsampled ERA5 grid
(latitude 70N -> 46.75N stored north->south, full longitude circle, 0.75 deg).
Coordinates: x eastward, y northward, with dx = R cos(phi) dlambda and
dy = R dphi, so for a field F(lambda, phi):
    dF/dx = F_lambda / (R cos phi),   dF/dy = F_phi / R.
"""
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from pde_diff.loss import VorticityLoss
from pde_diff.utils import GradientHelper

R = 6.37e6
LAT = 70.0 - 0.75 * np.arange(32)
LON = -180.0 + 0.75 * np.arange(480)
VARS = ["u", "v", "pv", "t", "z"]  # channel layout "(var lev)", 3 levels


def _mesh(dtype=torch.float64):
    lam = torch.deg2rad(torch.as_tensor(LON, dtype=dtype))
    phi = torch.deg2rad(torch.as_tensor(LAT, dtype=dtype))
    LAM, PHI = torch.meshgrid(lam, phi, indexing="ij")  # (lon, lat)
    return LAM[None, None], PHI[None, None]  # (1, 1, lon, lat)


@pytest.fixture(scope="module")
def gh():
    return GradientHelper(LAT, LON, radius=R)


def test_y_derivative_points_north_and_is_exact_for_linear_field(gh):
    _, PHI = _mesh()
    A = 1e-3
    phi_field = A * R * PHI  # increases northward with slope A per metre
    dy = gh.d_in_dy(phi_field)
    torch.testing.assert_close(dy, torch.full_like(dy, A), rtol=1e-9, atol=0)  # incl. edge rows


def test_y_derivatives_exact_for_quadratic_field(gh):
    _, PHI = _mesh()
    c = 1e-6
    y = R * PHI
    f = c * y**2
    torch.testing.assert_close(gh.d_in_dy(f), 2 * c * y, rtol=1e-9, atol=0)
    d2 = gh.d2_in_dy2(f)
    torch.testing.assert_close(d2, torch.full_like(d2, 2 * c), rtol=1e-6, atol=0)


def test_x_derivatives_periodic_and_scaled_by_cos_lat(gh):
    LAM, PHI = _mesh()
    f = torch.sin(LAM)
    rcos = R * torch.cos(PHI)
    torch.testing.assert_close(gh.d_in_dx(f), torch.cos(LAM) / rcos, rtol=0, atol=1e-4 / rcos.min().item())
    torch.testing.assert_close(gh.d2_in_dx2(f), -torch.sin(LAM) / rcos**2, rtol=0, atol=1e-4 / rcos.min().item() ** 2)


def test_vorticity_of_streamfunction_flow_equals_laplacian(gh):
    """u = -dpsi/dy, v = dpsi/dx (analytic)  =>  dv/dx - du/dy == lap(psi)."""
    LAM, PHI = _mesh()
    m, k = 3, 5.0
    psi = 1e7 * torch.sin(m * LAM) * torch.cos(k * PHI)
    u = -(1e7 * torch.sin(m * LAM) * (-k) * torch.sin(k * PHI)) / R
    v = (1e7 * m * torch.cos(m * LAM) * torch.cos(k * PHI)) / (R * torch.cos(PHI))
    zeta = gh.d_in_dx(v) - gh.d_in_dy(u)
    lap = gh.laplacian_horizontal(psi)
    scale = lap.abs().max()
    assert ((zeta - lap)[..., 1:-1].abs().max() / scale) < 1e-2
    assert ((zeta - lap).abs().max() / scale) < 5e-2  # edges: one-sided stencils, still close


def test_grid_validation():
    with pytest.raises(AssertionError):
        GradientHelper(LAT, LON[:-10], radius=R)  # longitude not covering the full circle
    with pytest.raises(AssertionError):
        GradientHelper(np.r_[LAT[:-1], 40.0], LON, radius=R)  # non-uniform latitude


def _loss():
    loss = VorticityLoss(OmegaConf.create({"name": "vorticity", "c_residual": 0.0, "device": "cpu"}))
    n = len(VARS) * 3
    loss.set_mean_and_std(np.zeros(n), np.ones(n), np.zeros(n), np.ones(n))  # identity normalisation
    return loss


def _pack(fields):
    """{var: (1,1,lon,lat) tensor} -> (1, 15, lon, lat) in "(var lev)" order."""
    LAM, _ = _mesh(torch.float32)
    zeros = torch.zeros_like(LAM)
    chans = [fields.get(var, zeros).float().expand_as(zeros) for var in VARS for _ in range(3)]
    return torch.cat(chans, dim=1)


def test_default_grid_matches_era5():
    loss = _loss()
    np.testing.assert_allclose(loss.grid_lat, LAT)
    np.testing.assert_allclose(loss.grid_lon, LON)


def _coriolis(loss, PHI):
    """Coriolis parameter in R2's geostrophic wind: the constant f0 (QG definition)."""
    return torch.full_like(PHI, float(loss.f0))


def test_geostrophic_residual_vanishes_for_geostrophic_wind():
    loss = _loss()
    LAM, PHI = _mesh()
    f = _coriolis(loss, PHI)
    m, k, A = 3, 5.0, 5e3  # A in m^2/s^2: winds of O(10 m/s)
    z = 5e4 + A * torch.sin(m * LAM) * torch.cos(k * PHI)
    dz_dx = A * m * torch.cos(m * LAM) * torch.cos(k * PHI) / (R * torch.cos(PHI))
    dz_dy = -A * k * torch.sin(m * LAM) * torch.sin(k * PHI) / R
    u, v = -dz_dy / f, dz_dx / f
    state = _pack({"u": u, "v": v, "z": z})
    r = loss.compute_residual_geostrophic_wind(torch.zeros_like(state), state, normalize=False)
    assert r.shape == (1, 2, 3, u.shape[-2], u.shape[-1] - 2 * loss.EDGE_ROWS)  # (b, component, lev, lon, interior lat)
    wind = (u.abs() + v.abs()).max().item()
    assert wind > 5.0
    assert r.abs().max().item() < 2e-2 * wind  # only discretisation + float32 error remains


def test_geostrophic_residual_has_correct_sign():
    """Wind opposite to geostrophic must give u_g - u = 2 u_g, not ~0."""
    loss = _loss()
    _, PHI = _mesh()
    A = 1e-3  # dPhi/dy > 0 (geopotential rising northward) -> u_g = -A/f(phi) (easterly)
    z = 5e4 + A * R * PHI
    u_g = -A / _coriolis(loss, PHI)  # varies with latitude through f(phi)
    state = _pack({"u": u_g, "z": z})
    r = loss.compute_residual_geostrophic_wind(torch.zeros_like(state), state, normalize=False)
    assert r.abs().max().item() < 1e-3 * u_g.abs().min().item()
    state_wrong = _pack({"u": -u_g, "z": z})
    r_wrong = loss.compute_residual_geostrophic_wind(torch.zeros_like(state_wrong), state_wrong, normalize=False)
    torch.testing.assert_close(r_wrong[:, 0], loss.interior(2 * u_g).expand_as(r_wrong[:, 0]), rtol=1e-3, atol=0, check_dtype=False)
    assert r_wrong[:, 1].abs().max().item() < 1e-3 * u_g.abs().min().item()


def test_geostrophic_residual_uses_f0():
    """R2 uses the QG f0: a wind balanced with f0 has zero residual at every latitude."""
    loss = _loss()
    _, PHI = _mesh()
    A = 1e-3
    z = 5e4 + A * R * PHI
    u_f0 = torch.full_like(PHI, -A / float(loss.f0))
    state = _pack({"u": u_f0, "z": z})
    assert loss.compute_residual_geostrophic_wind(torch.zeros_like(state), state, normalize=False).abs().max().item() < 1e-3


def test_vector_relative_residual_and_scale_only_normalisation():
    loss = _loss()
    _, PHI = _mesh()
    A = 1e-3
    z = 5e4 + A * R * PHI
    u_g = -A / _coriolis(loss, PHI)
    # wind = 1.1 * geostrophic -> |V - V_g| / (|V - V_g| + |V_g|) = 0.1 / 1.1
    state = _pack({"u": 1.1 * u_g, "z": z})
    rel = loss.compute_residual_geostrophic_wind(torch.zeros_like(state), state, relative="vector")
    torch.testing.assert_close(rel, torch.full_like(rel, 0.1 / 1.1), rtol=1e-3, atol=0)
    # R2 normalisation divides by the RMS only: exact balance must stay at 0
    loss.residual_stats = {"geostrophic_wind": {"mean": 3.0, "std": 4.0}}  # rms = 5
    balanced = _pack({"u": u_g, "z": z})
    r = loss.compute_residual_geostrophic_wind(torch.zeros_like(balanced), balanced)
    assert r.abs().max().item() < 1e-3
    r = loss.compute_residual_geostrophic_wind(torch.zeros_like(state), state)
    torch.testing.assert_close(r[:, 0], loss.interior(-0.1 * u_g / 5.0).expand_as(r[:, 0]), rtol=1e-3, atol=1e-6, check_dtype=False)


def test_geostrophic_loss_is_rotation_invariant():
    """du^2 + dv^2 depends only on |V - V_g|: the same ageostrophic error
    pointing east or north-east costs the same (the old (|du| + |dv|)^2
    charged the diagonal twice as much)."""
    loss = _loss()
    _, PHI = _mesh()
    z = torch.full_like(PHI, 5e4)  # flat geopotential: V_g = 0, so R2 = -V
    north_east = _pack({"u": torch.full_like(PHI, 2 ** -0.5), "v": torch.full_like(PHI, 2 ** -0.5), "z": z})
    east = _pack({"u": torch.ones_like(PHI), "z": z})
    sq = [(loss.compute_residual_geostrophic_wind(torch.zeros_like(s), s, normalize=False) ** 2).sum(1)
          for s in (east, north_east)]
    torch.testing.assert_close(sq[0], sq[1], rtol=1e-5, atol=1e-6)


def test_balance_ratio_bounds():
    from pde_diff.loss import balance_ratio
    a = torch.tensor([1.0, 2.0])
    num, den, n = balance_ratio([[a, -a]])  # perfect cancellation
    assert num == 0 and den == 6.0 and n == 2
    num, den, _ = balance_ratio([[a, a]])  # no cancellation
    assert num / den == pytest.approx(1.0)


def _two_states():
    """Two smooth, different atmospheric states (all 3 levels), packed as (1, 15, lon, lat)."""
    LAM, PHI = _mesh(torch.float32)
    lev = torch.tensor([1.0, 0.9, 0.8]).view(1, 3, 1, 1)  # vertical structure for the stretching term
    def state(shift):
        z = 5e4 * lev + 5e3 * torch.sin(3 * (LAM - shift)) * torch.cos(5 * PHI) * lev
        u = 10 + 5 * torch.cos(2 * (LAM - shift)) * torch.sin(4 * PHI) * lev
        v = 3 * torch.sin(2 * (LAM - shift)) * lev
        chans = {"u": u, "v": v, "z": z}
        zeros = torch.zeros_like(LAM)
        return torch.cat([chans[var][:, i:i + 1] if var in chans else zeros for var in VARS for i in range(3)], dim=1)
    return state(0.0), state(0.05)


@pytest.mark.parametrize("residual", ["compute_residual_planetary_vorticity", "compute_residual_vorticity_divergence"])
def test_time_centred_residuals_symmetric_in_time(residual):
    """Time-centred: swapping previous and current state flips only the tendency term,
    so R(p, c) - R(c, p) = 2 * tendency and R(p, c) + R(c, p) = 2 * centred advection."""
    loss = _loss()
    fn = getattr(loss, residual)
    s_p, s_c = _two_states()
    # identity normalisation: current = previous + change
    terms_pc = fn(s_p, s_c - s_p, normalize=False, return_terms=True)[0]
    terms_cp = fn(s_c, s_p - s_c, normalize=False, return_terms=True)[0]
    torch.testing.assert_close(terms_pc[0], -terms_cp[0])  # tendency flips sign
    for a, b in zip(terms_pc[1:], terms_cp[1:]):  # all other terms identical (centred in time)
        torch.testing.assert_close(a, b)
    # steady state: no tendency, residual = the (instantaneous) spatial terms
    steady = fn(s_p, torch.zeros_like(s_p), normalize=False, return_terms=True)[0]
    assert steady[0].abs().max().item() == 0.0


def test_residual_weightings():
    from pde_diff.loss import RESIDUAL_WEIGHTINGS, LOG_LIKELIHOOD_FLOOR
    r = torch.full((4, 3, 8, 8), 1.2)  # ERA5-like normalised R2 value
    var_t = torch.tensor([5.8e-5, 2e-3, 1e-2, 2e-2])
    ctx = {"var_t": var_t, "snr_weight": None, "intrinsic_var": 2.44}
    # intrinsic: bounded, 0.5 r^2 / (Sigma_t + sigma^2)
    loss, diag = RESIDUAL_WEIGHTINGS["intrinsic"](r, ctx)
    torch.testing.assert_close(loss[:, 0, 0, 0], 0.5 * 1.44 / (var_t + 2.44))
    assert diag == {} and loss.max().item() < 0.5
    # bastek (no clamp): 0.5 r^2 / Sigma_t, huge at small t
    loss, _ = RESIDUAL_WEIGHTINGS["bastek"](r, ctx)
    torch.testing.assert_close(loss[:, 0, 0, 0], 0.5 * 1.44 / var_t)
    # legacy clamped: capped at -floor, reports the clamped fraction.
    # Clamped where r^2 / Sigma_t > 55.3: at r = 1.2 every sample is clamped,
    # at r = 1.0 all but the largest Sigma_t (1 / 0.02 = 50).
    loss, diag = RESIDUAL_WEIGHTINGS["bastek_clamped"](r, ctx)
    assert loss.max().item() == pytest.approx(-LOG_LIKELIHOOD_FLOOR)
    assert diag["clamp_frac"].item() == pytest.approx(1.0)
    _, diag = RESIDUAL_WEIGHTINGS["bastek_clamped"](torch.ones_like(r), ctx)
    assert diag["clamp_frac"].item() == pytest.approx(0.75)


def test_intrinsic_weighting_default_and_gradient():
    """Default weighting is "intrinsic"; its gradient does not vanish for ERA5-like residuals."""
    loss = VorticityLoss(OmegaConf.create({"name": "vorticity", "c_residual": [0.0, 1e-2, 0.0], "device": "cpu"}))
    assert loss.residual_weighting == "intrinsic"
    n = len(VARS) * 3
    loss.set_mean_and_std(np.zeros(n), np.ones(n), np.zeros(n), np.ones(n))
    loss.residual_stats = {"geostrophic_wind": {"mean": 0.3, "std": 4.5}}
    assert all(loss.intrinsic_residual_variance(name) == 1.0 for name in ("pv", "geo", "vd"))
    s_p, s_c = _two_states()
    change = (s_c - s_p).clone().requires_grad_(True)
    x0_hat = torch.cat([s_p, change], dim=1)
    total = loss(model_out=change, target=torch.zeros_like(change), x0_hat=x0_hat, var=torch.full((1,), 1e-4))
    total.backward()
    assert "phys_clamp_frac_geo" not in loss.last_components
    assert loss.last_components["gradnorm_phys_geo"].item() > 0


def test_unknown_weighting_rejected():
    with pytest.raises(AssertionError):
        VorticityLoss(OmegaConf.create({"name": "vorticity", "c_residual": 0.0, "device": "cpu", "residual_weighting": "nope"}))


def test_residuals_exclude_edge_rows():
    """All residuals (and their balance terms) drop EDGE_ROWS latitude rows at
    each edge: an imbalance confined to the edge rows does not enter at all."""
    loss = _loss()
    assert loss.EDGE_ROWS == 1
    _, PHI = _mesh()
    z = torch.full_like(PHI, 5e4)  # V_g = 0
    u = torch.zeros_like(PHI)
    u[..., 0] = u[..., -1] = 10.0  # ageostrophic wind on the edge rows only
    state = _pack({"u": u, "z": z})
    r = loss.compute_residual_geostrophic_wind(torch.zeros_like(state), state, normalize=False)
    assert r.shape[-1] == len(LAT) - 2 and r.abs().max().item() == 0.0
    s_p, s_c = _two_states()
    for fn in (loss.compute_residual_planetary_vorticity, loss.compute_residual_geostrophic_wind,
               loss.compute_residual_vorticity_divergence):
        assert fn(s_p, s_c - s_p, normalize=False).shape[-1] == len(LAT) - 2
        assert all(t.shape[-1] == len(LAT) - 2 for group in fn(s_p, s_c - s_p, return_terms=True) for t in group)
    padded = loss.pad_edge_rows(r)
    assert padded.shape[-1] == len(LAT) and torch.isnan(padded[..., [0, -1]]).all()
