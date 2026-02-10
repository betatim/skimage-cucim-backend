"""Thin wrappers that forward to cucim.skimage.restoration."""

import cucim.skimage.restoration as cucim_restoration


def denoise_tv_chambolle(
    image, weight=0.1, eps=2.0e-4, max_num_iter=200, *, channel_axis=None
):
    """CuCIM-backed denoise_tv_chambolle; same signature as skimage.restoration.denoise_tv_chambolle."""
    return cucim_restoration.denoise_tv_chambolle(
        image,
        weight=weight,
        eps=eps,
        max_num_iter=max_num_iter,
        channel_axis=channel_axis,
    )


def richardson_lucy(image, psf, num_iter=50, clip=True, filter_epsilon=None):
    """CuCIM-backed richardson_lucy; same signature as skimage.restoration.richardson_lucy."""
    return cucim_restoration.richardson_lucy(
        image,
        psf,
        num_iter=num_iter,
        clip=clip,
        filter_epsilon=filter_epsilon,
    )
