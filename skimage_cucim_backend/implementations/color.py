"""Thin wrappers that forward to cucim.skimage.color."""

import cucim.skimage.color as cucim_color


def rgb2gray(rgb, *, channel_axis=-1):
    """CuCIM-backed rgb2gray; same signature as skimage.color.rgb2gray."""
    return cucim_color.rgb2gray(rgb, channel_axis=channel_axis)


def gray2rgb(image, *, channel_axis=-1):
    """CuCIM-backed gray2rgb; same signature as skimage.color.gray2rgb."""
    return cucim_color.gray2rgb(image, channel_axis=channel_axis)


def rgb2hsv(rgb, *, channel_axis=-1):
    """CuCIM-backed rgb2hsv; same signature as skimage.color.rgb2hsv."""
    return cucim_color.rgb2hsv(rgb, channel_axis=channel_axis)


def hsv2rgb(hsv, *, channel_axis=-1):
    """CuCIM-backed hsv2rgb; same signature as skimage.color.hsv2rgb."""
    return cucim_color.hsv2rgb(hsv, channel_axis=channel_axis)


def rgb2lab(rgb, illuminant="D65", observer="2", *, channel_axis=-1):
    """CuCIM-backed rgb2lab; same signature as skimage.color.rgb2lab."""
    return cucim_color.rgb2lab(
        rgb,
        illuminant=illuminant,
        observer=observer,
        channel_axis=channel_axis,
    )


def lab2rgb(lab, illuminant="D65", observer="2", *, channel_axis=-1):
    """CuCIM-backed lab2rgb; same signature as skimage.color.lab2rgb."""
    return cucim_color.lab2rgb(
        lab,
        illuminant=illuminant,
        observer=observer,
        channel_axis=channel_axis,
    )


def label2rgb(
    label,
    image=None,
    colors=None,
    alpha=0.3,
    bg_label=0,
    bg_color=(0, 0, 0),
    image_alpha=1,
    kind="overlay",
    *,
    saturation=0,
    channel_axis=-1,
):
    """CuCIM-backed label2rgb; same signature as skimage.color.label2rgb."""
    return cucim_color.label2rgb(
        label,
        image=image,
        colors=colors,
        alpha=alpha,
        bg_label=bg_label,
        bg_color=bg_color,
        image_alpha=image_alpha,
        kind=kind,
        saturation=saturation,
        channel_axis=channel_axis,
    )
