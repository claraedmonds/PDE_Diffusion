"""Helpers for working with the (optional) physics-residual terms in
VorticityLoss: reading their configured weights and wiring up the loss
module's normalization stats against a given dataset."""
from omegaconf import DictConfig, ListConfig

from pde_diff.model import DiffusionModel


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


def build_residual_loss_fn(model: DiffusionModel, dataset):
    """Wire up `model.loss_fn`'s normalization stats against `dataset`, if
    not already set, and return it (so residual computation matches how
    it's normalized during training/validation)."""
    loss_fn = model.loss_fn
    if not hasattr(loss_fn, "std") and hasattr(dataset, "means"):
        loss_fn.set_mean_and_std(dataset.means, dataset.stds, dataset.diff_means, dataset.diff_stds)
    return loss_fn
