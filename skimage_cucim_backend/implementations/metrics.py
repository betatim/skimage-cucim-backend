"""Thin wrappers that forward to cucim.skimage.metrics."""

import cupy as cp
import cucim.skimage.metrics as cucim_metrics


def mean_squared_error(image0, image1):
    """CuCIM-backed mean_squared_error; same signature as skimage.metrics.mean_squared_error."""
    return cucim_metrics.mean_squared_error(image0, image1)


def normalized_root_mse(image_true, image_test, *, normalization="euclidean"):
    """CuCIM-backed normalized_root_mse; same signature as skimage.metrics.normalized_root_mse."""
    return cucim_metrics.normalized_root_mse(
        image_true, image_test, normalization=normalization
    )


def peak_signal_noise_ratio(image_true, image_test, *, data_range=None):
    """CuCIM-backed peak_signal_noise_ratio; same signature as skimage.metrics.peak_signal_noise_ratio."""
    return cucim_metrics.peak_signal_noise_ratio(
        image_true, image_test, data_range=data_range
    )


def structural_similarity(
    im1, im2, *, win_size=None, gradient=False, data_range=None,
    channel_axis=None, gaussian_weights=False, full=False, **kwargs
):
    """CuCIM-backed structural_similarity; same signature as skimage.metrics.structural_similarity."""
    return cucim_metrics.structural_similarity(
        im1, im2, win_size=win_size, gradient=gradient, data_range=data_range,
        channel_axis=channel_axis, gaussian_weights=gaussian_weights,
        full=full, **kwargs
    )


def normalized_mutual_information(image0, image1, *, bins=100):
    """CuCIM-backed normalized_mutual_information; same signature as skimage.metrics.normalized_mutual_information."""
    return cucim_metrics.normalized_mutual_information(image0, image1, bins=bins)
