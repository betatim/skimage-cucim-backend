"""Thin wrappers that forward to cucim.skimage.exposure."""

import cucim.skimage.exposure as cucim_exposure


def equalize_hist(image, nbins=256, mask=None):
    """CuCIM-backed equalize_hist; same signature as skimage.exposure.equalize_hist."""
    return cucim_exposure.equalize_hist(image, nbins=nbins, mask=mask)


def equalize_adapthist(image, kernel_size=None, clip_limit=0.01, nbins=256):
    """CuCIM-backed equalize_adapthist; same signature as skimage.exposure.equalize_adapthist."""
    return cucim_exposure.equalize_adapthist(
        image,
        kernel_size=kernel_size,
        clip_limit=clip_limit,
        nbins=nbins,
    )


def match_histograms(image, reference, *, channel_axis=None):
    """CuCIM-backed match_histograms; same signature as skimage.exposure.match_histograms."""
    return cucim_exposure.match_histograms(image, reference, channel_axis=channel_axis)


def rescale_intensity(image, in_range="image", out_range="dtype"):
    """CuCIM-backed rescale_intensity; same signature as skimage.exposure.rescale_intensity."""
    return cucim_exposure.rescale_intensity(
        image, in_range=in_range, out_range=out_range
    )


def adjust_gamma(image, gamma=1, gain=1):
    """CuCIM-backed adjust_gamma; same signature as skimage.exposure.adjust_gamma."""
    return cucim_exposure.adjust_gamma(image, gamma=gamma, gain=gain)
