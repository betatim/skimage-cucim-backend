"""Thin wrappers that forward to cucim.skimage.feature."""

import numpy as np
import cupy as cp

import cucim.skimage.feature as cucim_feature


def canny(
    image,
    sigma=1.0,
    low_threshold=None,
    high_threshold=None,
    mask=None,
    use_quantiles=False,
    *,
    mode="constant",
    cval=0.0,
):
    """CuCIM-backed canny; same signature as skimage.feature.canny."""
    return cucim_feature.canny(
        image,
        sigma=sigma,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        mask=mask,
        use_quantiles=use_quantiles,
        mode=mode,
        cval=cval,
    )


def peak_local_max(
    image,
    min_distance=1,
    threshold_abs=None,
    threshold_rel=None,
    exclude_border=True,
    num_peaks=np.inf,
    footprint=None,
    labels=None,
    num_peaks_per_label=np.inf,
    p_norm=np.inf,
):
    """CuCIM-backed peak_local_max; same signature as skimage.feature.peak_local_max."""

    # CuCIM expects cp.inf for unbounded; translate from np.inf so comparisons work on device.
    def _to_cp_inf(x):
        if x is np.inf or (isinstance(x, (int, float)) and np.isinf(x)):
            return cp.inf
        return x

    num_peaks_arg = _to_cp_inf(num_peaks)
    num_peaks_per_label_arg = _to_cp_inf(num_peaks_per_label)
    p_norm_arg = _to_cp_inf(p_norm)
    return cucim_feature.peak_local_max(
        image,
        min_distance=min_distance,
        threshold_abs=threshold_abs,
        threshold_rel=threshold_rel,
        exclude_border=exclude_border,
        num_peaks=num_peaks_arg,
        footprint=footprint,
        labels=labels,
        num_peaks_per_label=num_peaks_per_label_arg,
        p_norm=p_norm_arg,
    )


def match_template(
    image, template, pad_input=False, mode="constant", constant_values=0
):
    """CuCIM-backed match_template; same signature as skimage.feature.match_template."""
    return cucim_feature.match_template(
        image,
        template,
        pad_input=pad_input,
        mode=mode,
        constant_values=constant_values,
    )
