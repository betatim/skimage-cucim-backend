"""Thin wrappers that forward to cucim.skimage.transform."""

import cucim.skimage.transform as cucim_transform


def resize(
    image,
    output_shape,
    order=None,
    mode='reflect',
    cval=0,
    clip=True,
    preserve_range=False,
    anti_aliasing=None,
    anti_aliasing_sigma=None,
):
    """CuCIM-backed resize; same signature as skimage.transform.resize."""
    return cucim_transform.resize(
        image,
        output_shape,
        order=order,
        mode=mode,
        cval=cval,
        clip=clip,
        preserve_range=preserve_range,
        anti_aliasing=anti_aliasing,
        anti_aliasing_sigma=anti_aliasing_sigma,
    )


def rescale(
    image,
    scale,
    order=None,
    mode='reflect',
    cval=0,
    clip=True,
    preserve_range=False,
    anti_aliasing=None,
    anti_aliasing_sigma=None,
    *,
    channel_axis=None,
):
    """CuCIM-backed rescale; same signature as skimage.transform.rescale."""
    return cucim_transform.rescale(
        image,
        scale,
        order=order,
        mode=mode,
        cval=cval,
        clip=clip,
        preserve_range=preserve_range,
        anti_aliasing=anti_aliasing,
        anti_aliasing_sigma=anti_aliasing_sigma,
        channel_axis=channel_axis,
    )
