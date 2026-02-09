"""Thin wrappers that forward to cucim.skimage.measure."""

import cucim.skimage.measure as cucim_measure


def label(label_image, background=None, return_num=False, connectivity=None):
    """CuCIM-backed label; same signature as skimage.measure.label.

    When return_num=True, returns (labels, num) with labels a CuPy array and num an int.
    """
    return cucim_measure.label(
        label_image,
        background=background,
        return_num=return_num,
        connectivity=connectivity,
    )
