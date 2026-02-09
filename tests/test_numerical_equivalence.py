"""Numerical equivalence: NumPy vs CuPy (dispatched) results match."""

import numpy as np
import pytest
from functools import partial

from skimage_cucim_backend._testing import identity_map, make_hist_for_otsu
from skimage_cucim_backend.information import SUPPORTED_FUNCTIONS
from skimage.util._backends import DispatchNotification, public_api_module

# Import so callables use the public API (dispatching applies when CuPy is passed).
import skimage.exposure as _exposure
import skimage.feature as _feature
import skimage.metrics as _metrics
import skimage.transform as _transform
import skimage.filters as _filters

# Suppress "dispatched to backend" warnings in this module; they add noise when we expect dispatch.
pytestmark = pytest.mark.filterwarnings(
    f"ignore::{DispatchNotification.__module__}.{DispatchNotification.__qualname__}"
)


_SEED = 42
_RTOL = 1e-5
_ATOL = 1e-8


def _to_numpy(result, cupy_module):
    """Convert result to numpy for comparison. result is from skimage call (np or cp)."""
    if isinstance(result, cupy_module.ndarray):
        return np.asarray(cupy_module.asnumpy(result))
    return np.asarray(result)


def _assert_result_is_cupy(result, cupy_module):
    """Assert that result (from a call with CuPy inputs) is a CuPy ndarray or tuple of CuPy ndarrays."""
    if isinstance(result, tuple):
        for r in result:
            assert isinstance(r, cupy_module.ndarray), (
                f"Expected cupy.ndarray, got {type(r)}"
            )
    else:
        assert isinstance(result, cupy_module.ndarray), (
            f"Expected cupy.ndarray, got {type(result)}"
        )


def _assert_allclose(ref, out, cupy_module, rtol=_RTOL, atol=_ATOL):
    """Compare ref (numpy result) and out (numpy or cupy result); convert out to numpy if needed."""
    ref_np = np.asarray(ref).ravel()
    out_np = _to_numpy(out, cupy_module).ravel()
    np.testing.assert_allclose(ref_np, out_np, rtol=rtol, atol=atol)


def _make_two_arrays(seed, xp):
    """Two arrays (im1, im2) of shape (7, 7), float64. Same logical data for np and cupy."""
    rng = np.random.default_rng(seed)
    a = xp.asarray(rng.random((7, 7), dtype=np.float64))
    b = xp.asarray(rng.random((7, 7), dtype=np.float64))
    return (a, b)


def _make_one_image(seed, xp):
    """One image (im,) of shape (7, 7), float64."""
    rng = np.random.default_rng(seed)
    im = xp.asarray(rng.random((7, 7), dtype=np.float64))
    return (im,)


def _make_image_and_template(seed, xp):
    """Image (7, 7) and template (3, 3) for match_template."""
    rng = np.random.default_rng(seed)
    image = xp.asarray(rng.random((7, 7), dtype=np.float64))
    template = xp.asarray(rng.random((3, 3), dtype=np.float64))
    return (image, template)


# ---- Metrics: f(im1, im2) ----
# List of (id, callable) for parametrize; callable(im1, im2) -> result.
METRICS_CALLABLES = [
    ("mean_squared_error", partial(_metrics.mean_squared_error)),
    ("normalized_root_mse", partial(_metrics.normalized_root_mse)),
    (
        "normalized_root_mse_euclidean",
        partial(_metrics.normalized_root_mse, normalization="euclidean"),
    ),
    (
        "normalized_root_mse_minmax",
        partial(_metrics.normalized_root_mse, normalization="min-max"),
    ),
    (
        "normalized_root_mse_mean",
        partial(_metrics.normalized_root_mse, normalization="mean"),
    ),
    ("peak_signal_noise_ratio", partial(_metrics.peak_signal_noise_ratio)),
    (
        "peak_signal_noise_ratio_data_range",
        partial(_metrics.peak_signal_noise_ratio, data_range=1.0),
    ),
    ("structural_similarity", partial(_metrics.structural_similarity, data_range=1.0)),
    (
        "structural_similarity_win_size",
        partial(_metrics.structural_similarity, data_range=1.0, win_size=7),
    ),
    ("normalized_mutual_information", partial(_metrics.normalized_mutual_information)),
    (
        "normalized_mutual_information_bins",
        partial(_metrics.normalized_mutual_information, bins=10),
    ),
]


def _integrate_via_image(im):
    """Wrapper: f(im) -> integrate(integral_image(im), ...). Uses public API so dispatch applies."""
    ii = _transform.integral_image(im)
    return _transform.integrate(ii, [(0, 0)], [(1, 1)])


# ---- Single image: f(im) - transforms and filters ----
# List of (id, callable) for parametrize; callable(im) -> result.
# Use partial(..., keyword=value) so the image stays as the first (only) positional.
# warp keeps a lambda because inverse_map is the second positional (no keyword in the API).
SINGLE_IMAGE_CALLABLES = [
    # Transforms
    ("resize", partial(_transform.resize, output_shape=(10, 10))),
    ("rescale", partial(_transform.rescale, scale=0.5)),
    ("rotate", partial(_transform.rotate, angle=45)),
    ("warp", lambda im: _transform.warp(im, identity_map, output_shape=(7, 7))),
    ("resize_local_mean", partial(_transform.resize_local_mean, output_shape=(4, 4))),
    ("downscale_local_mean", partial(_transform.downscale_local_mean, factors=(2, 2))),
    ("integral_image", partial(_transform.integral_image)),
    ("integrate", _integrate_via_image),
    ("pyramid_reduce", partial(_transform.pyramid_reduce)),
    ("pyramid_expand", partial(_transform.pyramid_expand)),
    ("swirl", partial(_transform.swirl)),
    ("warp_polar", partial(_transform.warp_polar, output_shape=(7, 7))),
    # Filters
    ("gaussian", partial(_filters.gaussian)),
    ("gaussian_sigma", partial(_filters.gaussian, sigma=2.0)),
    ("sobel", partial(_filters.sobel)),
    ("threshold_otsu_image", partial(_filters.threshold_otsu)),
    ("threshold_li", partial(_filters.threshold_li)),
    ("threshold_yen", partial(_filters.threshold_yen)),
    ("threshold_isodata", partial(_filters.threshold_isodata)),
    (
        "difference_of_gaussians",
        partial(_filters.difference_of_gaussians, low_sigma=2, high_sigma=10),
    ),
    ("prewitt", partial(_filters.prewitt)),
    ("scharr", partial(_filters.scharr)),
    ("median", partial(_filters.median)),
    ("laplace", partial(_filters.laplace)),
    ("roberts", partial(_filters.roberts)),
    ("unsharp_mask", partial(_filters.unsharp_mask)),
    # Feature (single image)
    ("canny", partial(_feature.canny)),
    ("peak_local_max", partial(_feature.peak_local_max)),
    # Exposure (single image)
    ("equalize_hist", partial(_exposure.equalize_hist)),
    ("equalize_adapthist", partial(_exposure.equalize_adapthist)),
    ("rescale_intensity", partial(_exposure.rescale_intensity)),
    ("adjust_gamma", partial(_exposure.adjust_gamma)),
]

# Feature: f(image, template) -> result (match_template only).
FEATURE_IMAGE_TEMPLATE_CALLABLES = [
    ("match_template", partial(_feature.match_template)),
]

# Exposure: f(image, reference) -> result (match_histograms only).
EXPOSURE_TWO_ARRAY_CALLABLES = [
    ("match_histograms", partial(_exposure.match_histograms)),
]

# Callables that don't unwrap to a single skimage function (lambda, wrapper); map scenario_id -> dispatch name.
_DISPATCH_NAME_FALLBACK = {
    "warp": "skimage.transform:warp",
    "integrate": "skimage.transform:integrate",
}


def _dispatch_name_from_callable(func):
    """Return dispatch name (skimage.module:func) for a callable that wraps a single skimage function, else None."""
    if isinstance(func, partial):
        func = func.func
    if not getattr(func, "__module__", "").startswith("skimage"):
        return None
    return f"{public_api_module(func)}:{func.__name__}"


def _numerical_equivalence_covered_dispatch_names():
    """Set of dispatch names (skimage.module:func) covered by at least one numerical equivalence scenario."""
    covered = set()
    for _scenario_id, func in METRICS_CALLABLES:
        name = _dispatch_name_from_callable(func)
        if name:
            covered.add(name)
    for scenario_id, func in SINGLE_IMAGE_CALLABLES:
        name = _dispatch_name_from_callable(func)
        if name:
            covered.add(name)
        elif scenario_id in _DISPATCH_NAME_FALLBACK:
            covered.add(_DISPATCH_NAME_FALLBACK[scenario_id])
    for _scenario_id, func in FEATURE_IMAGE_TEMPLATE_CALLABLES:
        name = _dispatch_name_from_callable(func)
        if name:
            covered.add(name)
    for _scenario_id, func in EXPOSURE_TWO_ARRAY_CALLABLES:
        name = _dispatch_name_from_callable(func)
        if name:
            covered.add(name)
    return covered


def test_every_supported_function_has_numerical_equivalence():
    """Every entry in SUPPORTED_FUNCTIONS is covered by at least one numerical equivalence test."""
    covered = _numerical_equivalence_covered_dispatch_names()
    missing = set(SUPPORTED_FUNCTIONS) - covered
    assert not missing, (
        f"SUPPORTED_FUNCTIONS entries not covered by numerical equivalence tests: {sorted(missing)}"
    )


@pytest.mark.cupy
@pytest.mark.parametrize(
    "scenario_id,func",
    METRICS_CALLABLES,
    ids=[p[0] for p in METRICS_CALLABLES],
)
def test_numerical_equivalence_f_im1_im2(scenario_id, func, cupy, require_cuda):
    """NumPy vs CuPy (dispatched) numerical equivalence for metrics f(im1, im2)."""
    np_arrays = _make_two_arrays(_SEED, np)
    cp_arrays = _make_two_arrays(_SEED, cupy)
    ref = func(*np_arrays)
    out = func(*cp_arrays)
    _assert_result_is_cupy(out, cupy)
    _assert_allclose(ref, out, cupy)


# Known numerical differences: integrate uses a different implementation (e.g. sum convention).
_INTEGRATE_ATOL = 0.2


@pytest.mark.cupy
@pytest.mark.parametrize(
    "scenario_id,func",
    SINGLE_IMAGE_CALLABLES,
    ids=[p[0] for p in SINGLE_IMAGE_CALLABLES],
)
def test_numerical_equivalence_f_im(scenario_id, func, cupy, require_cuda):
    """NumPy vs CuPy (dispatched) numerical equivalence for f(im): transforms and filters."""
    np_arrays = _make_one_image(_SEED, np)
    cp_arrays = _make_one_image(_SEED, cupy)
    ref = func(*np_arrays)
    out = func(*cp_arrays)
    _assert_result_is_cupy(out, cupy)
    atol = _INTEGRATE_ATOL if scenario_id == "integrate" else _ATOL
    _assert_allclose(ref, out, cupy, atol=atol)


@pytest.mark.cupy
@pytest.mark.parametrize(
    "scenario_id,func",
    FEATURE_IMAGE_TEMPLATE_CALLABLES,
    ids=[p[0] for p in FEATURE_IMAGE_TEMPLATE_CALLABLES],
)
def test_numerical_equivalence_f_im_template(scenario_id, func, cupy, require_cuda):
    """NumPy vs CuPy (dispatched) numerical equivalence for feature f(image, template)."""
    np_arrays = _make_image_and_template(_SEED, np)
    cp_arrays = _make_image_and_template(_SEED, cupy)
    ref = func(*np_arrays)
    out = func(*cp_arrays)
    _assert_result_is_cupy(out, cupy)
    _assert_allclose(ref, out, cupy)


@pytest.mark.cupy
@pytest.mark.parametrize(
    "scenario_id,func",
    EXPOSURE_TWO_ARRAY_CALLABLES,
    ids=[p[0] for p in EXPOSURE_TWO_ARRAY_CALLABLES],
)
def test_numerical_equivalence_f_im_reference(scenario_id, func, cupy, require_cuda):
    """NumPy vs CuPy (dispatched) numerical equivalence for exposure f(image, reference)."""
    np_arrays = _make_two_arrays(_SEED, np)
    cp_arrays = _make_two_arrays(_SEED, cupy)
    ref = func(*np_arrays)
    out = func(*cp_arrays)
    _assert_result_is_cupy(out, cupy)
    _assert_allclose(ref, out, cupy)


@pytest.mark.cupy
def test_numerical_equivalence_threshold_otsu_hist(cupy, require_cuda):
    """NumPy vs CuPy numerical equivalence for threshold_otsu(hist=(counts, bin_centers))."""
    np_hist = make_hist_for_otsu(np, n=10)
    cp_hist = make_hist_for_otsu(cupy, n=10)
    ref = _filters.threshold_otsu(hist=np_hist)
    out = _filters.threshold_otsu(hist=cp_hist)
    _assert_result_is_cupy(out, cupy)
    _assert_allclose(ref, out, cupy)


@pytest.mark.cupy
def test_numerical_equivalence_threshold_yen_hist(cupy, require_cuda):
    """NumPy vs CuPy numerical equivalence for threshold_yen(hist=(counts, bin_centers))."""
    np_hist = make_hist_for_otsu(np, n=10)
    cp_hist = make_hist_for_otsu(cupy, n=10)
    ref = _filters.threshold_yen(hist=np_hist)
    out = _filters.threshold_yen(hist=cp_hist)
    _assert_result_is_cupy(out, cupy)
    _assert_allclose(ref, out, cupy)


@pytest.mark.cupy
def test_numerical_equivalence_threshold_isodata_hist(cupy, require_cuda):
    """NumPy vs CuPy numerical equivalence for threshold_isodata(hist=(counts, bin_centers))."""
    np_hist = make_hist_for_otsu(np, n=10)
    cp_hist = make_hist_for_otsu(cupy, n=10)
    ref = _filters.threshold_isodata(hist=np_hist)
    out = _filters.threshold_isodata(hist=cp_hist)
    _assert_result_is_cupy(out, cupy)
    _assert_allclose(ref, out, cupy)
