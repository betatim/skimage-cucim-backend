"""Thin wrappers that forward to cucim.skimage.morphology."""

import cucim.skimage.morphology as cucim_morphology

# Sentinel for deprecated min_size/area_threshold; backend uses max_size when this is passed.
_MISSING = object()


def remove_small_objects(
    ar, min_size=_MISSING, connectivity=1, *, max_size=63, out=None
):
    """CuCIM-backed remove_small_objects; same signature as skimage.morphology.remove_small_objects.

    CuCIM uses min_size (remove objects smaller than min_size); scikit-image uses
    max_size (remove objects with area <= max_size). We translate: use min_size if
    passed (deprecated API), else min_size = max_size + 1 for CuCIM.
    """
    cucim_min_size = (max_size + 1) if min_size is _MISSING else min_size
    return cucim_morphology.remove_small_objects(
        ar,
        min_size=cucim_min_size,
        connectivity=connectivity,
        out=out,
    )


def remove_small_holes(
    ar, area_threshold=_MISSING, connectivity=1, *, max_size=63, out=None
):
    """CuCIM-backed remove_small_holes; same signature as skimage.morphology.remove_small_holes.

    CuCIM uses area_threshold (fill holes with area <= area_threshold); scikit-image
    uses max_size (fill holes with area <= max_size). We pass area_threshold=max_size
    when using new API, else area_threshold from deprecated param.
    """
    cucim_area_threshold = max_size if area_threshold is _MISSING else area_threshold
    return cucim_morphology.remove_small_holes(
        ar,
        area_threshold=cucim_area_threshold,
        connectivity=connectivity,
        out=out,
    )


def erosion(image, footprint=None, out=None, *, mode="reflect", cval=0.0):
    """CuCIM-backed erosion; same signature as skimage.morphology.erosion."""
    return cucim_morphology.erosion(
        image,
        footprint=footprint,
        out=out,
        mode=mode,
        cval=cval,
    )


def dilation(image, footprint=None, out=None, *, mode="reflect", cval=0.0):
    """CuCIM-backed dilation; same signature as skimage.morphology.dilation."""
    return cucim_morphology.dilation(
        image,
        footprint=footprint,
        out=out,
        mode=mode,
        cval=cval,
    )


def opening(image, footprint=None, out=None, *, mode="reflect", cval=0.0):
    """CuCIM-backed opening; same signature as skimage.morphology.opening."""
    return cucim_morphology.opening(
        image,
        footprint=footprint,
        out=out,
        mode=mode,
        cval=cval,
    )


def closing(image, footprint=None, out=None, *, mode="reflect", cval=0.0):
    """CuCIM-backed closing; same signature as skimage.morphology.closing."""
    return cucim_morphology.closing(
        image,
        footprint=footprint,
        out=out,
        mode=mode,
        cval=cval,
    )


def white_tophat(image, footprint=None, out=None, *, mode="reflect", cval=0.0):
    """CuCIM-backed white_tophat; same signature as skimage.morphology.white_tophat."""
    return cucim_morphology.white_tophat(
        image,
        footprint=footprint,
        out=out,
        mode=mode,
        cval=cval,
    )


def black_tophat(image, footprint=None, out=None, *, mode="reflect", cval=0.0):
    """CuCIM-backed black_tophat; same signature as skimage.morphology.black_tophat."""
    return cucim_morphology.black_tophat(
        image,
        footprint=footprint,
        out=out,
        mode=mode,
        cval=cval,
    )
