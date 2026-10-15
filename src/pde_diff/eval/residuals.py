"""Helpers for working with the (optional) physics-residual terms in
VorticityLoss: reading their configured weights and wiring up the loss
evaluator's normalization stats against a given dataset."""
from omegaconf import DictConfig, ListConfig, OmegaConf

from pde_diff.loss import VorticityLoss


def get_residual_weights(model_cfg: DictConfig) -> list[float]:
    """Read `cfg.loss.c_residual`, normalized to a per-residual-fn list the
    same way `VorticityLoss.__init__` does (scalar broadcasts to 3 terms:
    planetary vorticity, geostrophic wind, vorticity divergence)."""
    c_residual = model_cfg.loss.get("c_residual", None)
    n_residuals = 3
    if c_residual is None:
        return [0.0] * n_residuals
    if isinstance(c_residual, (list, ListConfig)):
        return list(c_residual)
    return [c_residual] * n_residuals


def build_residual_evaluator(dataset, device) -> VorticityLoss:
    """Standalone VorticityLoss for computing the R1/R2/R3 physics residuals
    on any model's forecasts, independent of the loss the model was trained
    with (so MSE-only and physics-informed models are evaluated identically).
    Normalization stats come from `dataset`."""
    loss_fn = VorticityLoss(OmegaConf.create({"name": "vorticity", "c_residual": 0.0, "device": str(device)}))
    loss_fn.set_mean_and_std(dataset.means, dataset.stds, dataset.diff_means, dataset.diff_stds)
    return loss_fn
