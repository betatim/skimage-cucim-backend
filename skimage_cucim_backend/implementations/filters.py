"""Thin wrappers that forward to cucim.skimage.filters."""

import cucim.skimage.filters as cucim_filters


def gaussian(
    image,
    sigma=1.0,
    *,
    mode="nearest",
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


def sobel(image, mask=None, *, axis=None, mode="reflect", cval=0.0):
    """CuCIM-backed sobel; same signature as skimage.filters.sobel."""
    return cucim_filters.sobel(image, mask=mask, axis=axis, mode=mode, cval=cval)


def threshold_otsu(image=None, nbins=256, *, hist=None):
    """CuCIM-backed threshold_otsu; same signature as skimage.filters.threshold_otsu."""
    return cucim_filters.threshold_otsu(image, nbins=nbins, hist=hist)


def threshold_li(image, *, tolerance=None, initial_guess=None, iter_callback=None):
    """CuCIM-backed threshold_li; same signature as skimage.filters.threshold_li."""
    return cucim_filters.threshold_li(
        image,
        tolerance=tolerance,
        initial_guess=initial_guess,
        iter_callback=iter_callback,
    )


def threshold_yen(image=None, nbins=256, *, hist=None):
    """CuCIM-backed threshold_yen; same signature as skimage.filters.threshold_yen."""
    return cucim_filters.threshold_yen(image, nbins=nbins, hist=hist)


def threshold_isodata(image=None, nbins=256, return_all=False, *, hist=None):
    """CuCIM-backed threshold_isodata; same signature as skimage.filters.threshold_isodata."""
    return cucim_filters.threshold_isodata(
        image, nbins=nbins, return_all=return_all, hist=hist
    )


def difference_of_gaussians(
    image,
    low_sigma,
    high_sigma=None,
    *,
    mode="nearest",
    cval=0,
    channel_axis=None,
    truncate=4.0,
):
    """CuCIM-backed difference_of_gaussians; same signature as skimage.filters.difference_of_gaussians."""
    return cucim_filters.difference_of_gaussians(
        image,
        low_sigma,
        high_sigma=high_sigma,
        mode=mode,
        cval=cval,
        channel_axis=channel_axis,
        truncate=truncate,
    )


def prewitt(image, mask=None, *, axis=None, mode="reflect", cval=0.0):
    """CuCIM-backed prewitt; same signature as skimage.filters.prewitt."""
    return cucim_filters.prewitt(image, mask=mask, axis=axis, mode=mode, cval=cval)


def scharr(image, mask=None, *, axis=None, mode="reflect", cval=0.0):
    """CuCIM-backed scharr; same signature as skimage.filters.scharr."""
    return cucim_filters.scharr(image, mask=mask, axis=axis, mode=mode, cval=cval)


def median(
    image, footprint=None, out=None, mode="nearest", cval=0.0, behavior="ndimage"
):
    """CuCIM-backed median; same signature as skimage.filters.median."""
    return cucim_filters.median(
        image,
        footprint=footprint,
        out=out,
        mode=mode,
        cval=cval,
        behavior=behavior,
    )
