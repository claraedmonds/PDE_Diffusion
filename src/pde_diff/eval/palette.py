"""Generated (non-hardcoded) color palette for comparing an arbitrary number
of models in eval plots (PSD overlays, forecast-vs-step curves, loss-curve
comparisons)."""
import matplotlib.pyplot as plt


def get_model_color(idx: int, n_total: int):
    """Return a color for model `idx` out of `n_total` models being compared.

    Uses tab10 for up to 10 models (distinct, high-contrast categorical
    colors) and falls back to a continuous colormap for more.
    """
    if n_total <= 10:
        return plt.get_cmap("tab10")(idx % 10)
    return plt.get_cmap("viridis")(idx / max(1, n_total - 1))
