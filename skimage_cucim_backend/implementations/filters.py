"""Thin wrappers that forward to cucim.skimage.filters."""

import cucim.skimage.filters as cucim_filters


def gaussian(
    image,
    sigma=1.0,
    *,
    mode='nearest',
    cval=0,
    preserve_range=False,
    truncate=4.0,
    channel_axis=None,
    out=None,
):
    """CuCIM-backed gaussian; same signature as skimage.filters.gaussian."""
    return cucim_filters.gaussian(
        image,
        sigma=sigma,
        mode=mode,
        cval=cval,
        preserve_range=preserve_range,
        truncate=truncate,
        channel_axis=channel_axis,
        out=out,
    )


def sobel(image, mask=None, *, axis=None, mode='reflect', cval=0.0):
    """CuCIM-backed sobel; same signature as skimage.filters.sobel."""
    return cucim_filters.sobel(
        image, mask=mask, axis=axis, mode=mode, cval=cval
    )


def threshold_otsu(image=None, nbins=256, *, hist=None):
    """CuCIM-backed threshold_otsu; same signature as skimage.filters.threshold_otsu."""
    return cucim_filters.threshold_otsu(
        image, nbins=nbins, hist=hist
    )
