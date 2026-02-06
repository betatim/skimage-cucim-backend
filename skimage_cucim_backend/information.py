"""Backend info: fast, no heavy imports."""

from skimage.util._backends import BackendInformation

SUPPORTED_FUNCTIONS = [
    "skimage.metrics:mean_squared_error",
    "skimage.metrics:normalized_root_mse",
    "skimage.metrics:peak_signal_noise_ratio",
    "skimage.metrics:structural_similarity",
]


def info():
    """Return backend metadata. Must be fast; avoid importing cucim/cupy here."""
    return BackendInformation(SUPPORTED_FUNCTIONS)
