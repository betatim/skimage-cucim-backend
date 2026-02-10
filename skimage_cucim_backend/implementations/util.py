"""Thin wrappers that forward to cucim.skimage.util."""

import cucim.skimage.util as cucim_util


def invert(image, signed_float=False):
    """CuCIM-backed invert; same signature as skimage.util.invert."""
    return cucim_util.invert(image, signed_float=signed_float)


def compare_images(image0, image1, *, method="diff", n_tiles=(8, 8)):
    """CuCIM-backed compare_images; same signature as skimage.util.compare_images."""
    return cucim_util.compare_images(image0, image1, method=method, n_tiles=n_tiles)


def montage(
    arr_in,
    fill="mean",
    rescale_intensity=False,
    grid_shape=None,
    padding_width=0,
    *,
    channel_axis=None,
):
    """CuCIM-backed montage; same signature as skimage.util.montage."""
    return cucim_util.montage(
        arr_in,
        fill=fill,
        rescale_intensity=rescale_intensity,
        grid_shape=grid_shape,
        padding_width=padding_width,
        channel_axis=channel_axis,
        square_grid_default=True,
    )


def crop(ar, crop_width, copy=False, order="K"):
    """CuCIM-backed crop; same signature as skimage.util.crop."""
    return cucim_util.crop(ar, crop_width, copy=copy, order=order)


def random_noise(image, mode="gaussian", rng=None, clip=True, **kwargs):
    """CuCIM-backed random_noise; same signature as skimage.util.random_noise."""
    return cucim_util.random_noise(image, mode=mode, rng=rng, clip=clip, **kwargs)


def view_as_blocks(arr_in, block_shape):
    """CuCIM-backed view_as_blocks; same signature as skimage.util.view_as_blocks."""
    return cucim_util.view_as_blocks(arr_in, block_shape)


def view_as_windows(arr_in, window_shape, step=1):
    """CuCIM-backed view_as_windows; same signature as skimage.util.view_as_windows."""
    return cucim_util.view_as_windows(arr_in, window_shape, step=step)
