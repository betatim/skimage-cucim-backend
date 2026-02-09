"""Backend info: fast, no heavy imports."""

from skimage.util._backends import BackendInformation

SUPPORTED_FUNCTIONS = [
    "skimage.metrics:mean_squared_error",
    "skimage.metrics:normalized_root_mse",
    "skimage.metrics:peak_signal_noise_ratio",
    "skimage.metrics:structural_similarity",
    "skimage.metrics:normalized_mutual_information",
    "skimage.transform:resize",
    "skimage.transform:rescale",
    "skimage.transform:rotate",
    "skimage.transform:warp",
    "skimage.transform:resize_local_mean",
    "skimage.transform:downscale_local_mean",
    "skimage.transform:integral_image",
    "skimage.transform:integrate",
    "skimage.transform:pyramid_reduce",
    "skimage.transform:pyramid_expand",
    "skimage.transform:swirl",
    "skimage.transform:warp_polar",
    "skimage.filters:gaussian",
    "skimage.filters:sobel",
    "skimage.filters:threshold_otsu",
    "skimage.filters:threshold_li",
    "skimage.filters:threshold_yen",
    "skimage.filters:threshold_isodata",
    "skimage.filters:difference_of_gaussians",
    "skimage.filters:prewitt",
    "skimage.filters:scharr",
    "skimage.filters:median",
    "skimage.filters:laplace",
    "skimage.filters:roberts",
    "skimage.filters:unsharp_mask",
    "skimage.feature:canny",
    "skimage.feature:peak_local_max",
    "skimage.feature:match_template",
    "skimage.exposure:equalize_hist",
    "skimage.exposure:equalize_adapthist",
    "skimage.exposure:match_histograms",
    "skimage.exposure:rescale_intensity",
    "skimage.exposure:adjust_gamma",
]


def info():
    """Return backend metadata. Must be fast; avoid importing cucim/cupy here."""
    return BackendInformation(SUPPORTED_FUNCTIONS)
