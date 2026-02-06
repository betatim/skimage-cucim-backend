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


def rotate(
    image,
    angle,
    resize=False,
    center=None,
    order=None,
    mode='constant',
    cval=0,
    clip=True,
    preserve_range=False,
):
    """CuCIM-backed rotate; same signature as skimage.transform.rotate."""
    return cucim_transform.rotate(
        image,
        angle,
        resize=resize,
        center=center,
        order=order,
        mode=mode,
        cval=cval,
        clip=clip,
        preserve_range=preserve_range,
    )


def warp(
    image,
    inverse_map,
    map_args=None,
    output_shape=None,
    order=None,
    mode='constant',
    cval=0.0,
    clip=True,
    preserve_range=False,
):
    """CuCIM-backed warp; same signature as skimage.transform.warp."""
    return cucim_transform.warp(
        image,
        inverse_map,
        map_args=map_args,
        output_shape=output_shape,
        order=order,
        mode=mode,
        cval=cval,
        clip=clip,
        preserve_range=preserve_range,
    )


def resize_local_mean(
    image, output_shape, grid_mode=True, preserve_range=False, *, channel_axis=None
):
    """CuCIM-backed resize_local_mean; same signature as skimage.transform.resize_local_mean."""
    return cucim_transform.resize_local_mean(
        image,
        output_shape,
        grid_mode=grid_mode,
        preserve_range=preserve_range,
        channel_axis=channel_axis,
    )


def downscale_local_mean(image, factors, cval=0, clip=True):
    """CuCIM-backed downscale_local_mean; same signature as skimage.transform.downscale_local_mean."""
    return cucim_transform.downscale_local_mean(
        image, factors, cval=cval, clip=clip
    )


def integral_image(image, *, dtype=None):
    """CuCIM-backed integral_image; same signature as skimage.transform.integral_image."""
    return cucim_transform.integral_image(image, dtype=dtype)


def integrate(ii, start, end):
    """CuCIM-backed integrate; same signature as skimage.transform.integrate."""
    return cucim_transform.integrate(ii, start, end)


def pyramid_reduce(
    image,
    downscale=2,
    sigma=None,
    order=1,
    mode='reflect',
    cval=0,
    preserve_range=False,
    *,
    channel_axis=None,
):
    """CuCIM-backed pyramid_reduce; same signature as skimage.transform.pyramid_reduce."""
    return cucim_transform.pyramid_reduce(
        image,
        downscale=downscale,
        sigma=sigma,
        order=order,
        mode=mode,
        cval=cval,
        preserve_range=preserve_range,
        channel_axis=channel_axis,
    )


def pyramid_expand(
    image,
    upscale=2,
    sigma=None,
    order=1,
    mode='reflect',
    cval=0,
    preserve_range=False,
    *,
    channel_axis=None,
):
    """CuCIM-backed pyramid_expand; same signature as skimage.transform.pyramid_expand."""
    return cucim_transform.pyramid_expand(
        image,
        upscale=upscale,
        sigma=sigma,
        order=order,
        mode=mode,
        cval=cval,
        preserve_range=preserve_range,
        channel_axis=channel_axis,
    )


def swirl(
    image,
    center=None,
    strength=1,
    radius=100,
    rotation=0,
    output_shape=None,
    order=None,
    mode='reflect',
    cval=0,
    clip=True,
    preserve_range=False,
):
    """CuCIM-backed swirl; same signature as skimage.transform.swirl."""
    return cucim_transform.swirl(
        image,
        center=center,
        strength=strength,
        radius=radius,
        rotation=rotation,
        output_shape=output_shape,
        order=order,
        mode=mode,
        cval=cval,
        clip=clip,
        preserve_range=preserve_range,
    )


def warp_polar(
    image,
    center=None,
    *,
    radius=None,
    output_shape=None,
    scaling='linear',
    channel_axis=None,
    **kwargs,
):
    """CuCIM-backed warp_polar; same signature as skimage.transform.warp_polar."""
    return cucim_transform.warp_polar(
        image,
        center=center,
        radius=radius,
        output_shape=output_shape,
        scaling=scaling,
        channel_axis=channel_axis,
        **kwargs,
    )


