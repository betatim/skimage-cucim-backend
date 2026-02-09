"""Thin wrappers that forward to cucim.skimage.segmentation."""

import cucim.skimage.segmentation as cucim_segmentation


def clear_border(labels, buffer_size=0, bgval=0, mask=None, *, out=None):
    """CuCIM-backed clear_border; same signature as skimage.segmentation.clear_border."""
    return cucim_segmentation.clear_border(
        labels,
        buffer_size=buffer_size,
        bgval=bgval,
        mask=mask,
        out=out,
    )


def expand_labels(label_image, distance=1, spacing=1):
    """CuCIM-backed expand_labels; same signature as skimage.segmentation.expand_labels."""
    return cucim_segmentation.expand_labels(
        label_image,
        distance=distance,
        spacing=spacing,
    )


def find_boundaries(label_img, connectivity=1, mode="thick", background=0):
    """CuCIM-backed find_boundaries; same signature as skimage.segmentation.find_boundaries."""
    return cucim_segmentation.find_boundaries(
        label_img,
        connectivity=connectivity,
        mode=mode,
        background=background,
    )
