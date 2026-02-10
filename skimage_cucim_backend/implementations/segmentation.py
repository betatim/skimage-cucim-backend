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


def join_segmentations(s1, s2, return_mapping=False):
    """CuCIM-backed join_segmentations; same signature as skimage.segmentation.join_segmentations."""
    return cucim_segmentation.join_segmentations(s1, s2, return_mapping=return_mapping)


def relabel_sequential(label_field, offset=1):
    """CuCIM-backed relabel_sequential; same signature as skimage.segmentation.relabel_sequential."""
    return cucim_segmentation.relabel_sequential(label_field, offset=offset)


def mark_boundaries(
    image,
    label_img,
    color=(1, 1, 0),
    outline_color=None,
    mode="outer",
    background_label=0,
):
    """CuCIM-backed mark_boundaries; same signature as skimage.segmentation.mark_boundaries."""
    return cucim_segmentation.mark_boundaries(
        image,
        label_img,
        color=color,
        outline_color=outline_color,
        mode=mode,
        background_label=background_label,
        order=3,
    )
