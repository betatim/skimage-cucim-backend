"""Per-function invariant tests: CuPy in->out, no NumPy, shape match."""

import numpy as np
import pytest

from skimage_cucim_backend._testing import identity_map, make_hist_for_otsu
from skimage_cucim_backend.implementations import can_has, get_implementation
from skimage_cucim_backend.information import SUPPORTED_FUNCTIONS


def _skimage_reference_result(name, np_arrays, args, kwargs):
    """Call the scikit-image implementation with NumPy arrays; return the result.

    Used to derive expected shape and ndim for backend result comparison.
    np_arrays: (a, b) for metrics (two-array functions), or (a,) for transform (image + args).
    """
    _, rest = name.split(".", maxsplit=1)
    module_path, func_name = rest.rsplit(":", maxsplit=1)
    if module_path == "metrics":
        import skimage.metrics as mod
        func = getattr(mod, func_name)
        return func(np_arrays[0], np_arrays[1], *args, **kwargs)
    if module_path == "transform":
        import skimage.transform as mod
        func = getattr(mod, func_name)
        if func_name == "integrate":
            ii = mod.integral_image(np_arrays[0])
            return func(ii, *args, **kwargs)
        return func(np_arrays[0], *args, **kwargs)
    if module_path == "filters":
        import skimage.filters as mod
        func = getattr(mod, func_name)
        return func(np_arrays[0], *args, **kwargs)
    raise LookupError(f"No reference implementation for: {name}")


def _expected_shape_and_ndim(reference_result):
    """Return (shape, ndim) from a scikit-image return value (array or scalar)."""
    arr = np.asarray(reference_result)
    return arr.shape, arr.ndim


# One or more (name, args, kwargs) per supported function, covering several
# combinations of options where a function has multiple arguments or values.
# Arrays a, b are created in each test; these supply extra args/kwargs.
INVARIANT_CALL_PARAMS = [
    # mean_squared_error: no optional args
    ("skimage.metrics:mean_squared_error", (), {}),
    # normalized_root_mse: normalization in {'euclidean', 'min-max', 'mean'}
    ("skimage.metrics:normalized_root_mse", (), {}),
    ("skimage.metrics:normalized_root_mse", (), {"normalization": "euclidean"}),
    ("skimage.metrics:normalized_root_mse", (), {"normalization": "min-max"}),
    ("skimage.metrics:normalized_root_mse", (), {"normalization": "mean"}),
    # peak_signal_noise_ratio: data_range optional
    ("skimage.metrics:peak_signal_noise_ratio", (), {}),
    ("skimage.metrics:peak_signal_noise_ratio", (), {"data_range": 1.0}),
    # structural_similarity: pass data_range explicitly (recommended for float images in both skimage and CuCIM)
    ("skimage.metrics:structural_similarity", (), {"data_range": 1.0}),
    ("skimage.metrics:structural_similarity", (), {"data_range": 1.0, "win_size": 7}),
    # normalized_mutual_information: bins optional
    ("skimage.metrics:normalized_mutual_information", (), {}),
    ("skimage.metrics:normalized_mutual_information", (), {"bins": 10}),
]

# (name, args, kwargs) for filter functions: call impl(image, *args, **kwargs).
# threshold_otsu: first with image; second with hist only (kwargs["hist"] = (counts, bin_centers)).
FILTER_INVARIANT_CALL_PARAMS = [
    ("skimage.filters:gaussian", (), {}),
    ("skimage.filters:gaussian", (), {"sigma": 2.0}),
    ("skimage.filters:sobel", (), {}),
    ("skimage.filters:threshold_otsu", (), {}),
    ("skimage.filters:threshold_otsu", (), {"hist": make_hist_for_otsu(np)}),
    ("skimage.filters:difference_of_gaussians", (2, 10), {}),
    ("skimage.filters:prewitt", (), {}),
    ("skimage.filters:scharr", (), {}),
    ("skimage.filters:median", (), {}),
]

# (name, args, kwargs) for transform functions: call impl(image, *args, **kwargs); returns array.
TRANSFORM_INVARIANT_CALL_PARAMS = [
    ("skimage.transform:resize", ((10, 10),), {}),
    ("skimage.transform:rescale", (0.5,), {}),
    ("skimage.transform:rotate", (45,), {}),
    ("skimage.transform:warp", (identity_map,), {"output_shape": (7, 7)}),
    ("skimage.transform:resize_local_mean", ((4, 4),), {}),
    ("skimage.transform:downscale_local_mean", ((2, 2),), {}),
    ("skimage.transform:integral_image", (), {}),
    ("skimage.transform:integrate", ([(0, 0)], [(1, 1)]), {}),
    ("skimage.transform:pyramid_reduce", (), {}),
    ("skimage.transform:pyramid_expand", (), {}),
    ("skimage.transform:swirl", (), {}),
    ("skimage.transform:warp_polar", (), {"output_shape": (7, 7)}),
]

def test_all_supported_functions_covered_in_invariant_call_params():
    """Every SUPPORTED_FUNCTIONS entry appears at least once in invariant call params."""
    param_names = (
        {name for name, _, _ in INVARIANT_CALL_PARAMS}
        | {name for name, _, _ in FILTER_INVARIANT_CALL_PARAMS}
        | {name for name, _, _ in TRANSFORM_INVARIANT_CALL_PARAMS}
    )
    for fn in SUPPORTED_FUNCTIONS:
        assert fn in param_names, (
            f"{fn} missing from INVARIANT_CALL_PARAMS, "
            f"FILTER_INVARIANT_CALL_PARAMS, or TRANSFORM_INVARIANT_CALL_PARAMS"
        )


# ---- Invariant 1: CuPy in -> CuPy out (0-dim array for these metrics) ----
@pytest.mark.cupy
@pytest.mark.parametrize("name,args,kwargs", INVARIANT_CALL_PARAMS)
def test_invariant_cupy_in_cupy_out_scalar(name, args, kwargs, cupy, minimal_cupy_arrays_2d):
    """Backend returns 0-dim CuPy array for these metrics when CuPy in."""
    a, b = minimal_cupy_arrays_2d
    impl = get_implementation(name)
    result = impl(a, b, *args, **kwargs)
    assert isinstance(result, cupy.ndarray) and result.ndim == 0


# ---- Invariant 2: Does not accept NumPy input ----
@pytest.mark.parametrize("name,args,kwargs", INVARIANT_CALL_PARAMS)
def test_invariant_no_numpy(name, args, kwargs):
    """can_has returns False for NumPy array inputs (two-array metrics)."""
    a = np.array([[0.0, 0.5], [0.2, 0.8]], dtype=np.float64)
    b = np.array([[0.1, 0.4], [0.3, 0.9]], dtype=np.float64)
    assert not can_has(name, a, b, *args, **kwargs)


@pytest.mark.parametrize("name,args,kwargs", TRANSFORM_INVARIANT_CALL_PARAMS)
def test_invariant_no_numpy_transform(name, args, kwargs):
    """can_has returns False for NumPy array inputs (transform: single image + args)."""
    image = np.array([[0.0, 0.5], [0.2, 0.8]], dtype=np.float64)
    assert not can_has(name, image, *args, **kwargs)


@pytest.mark.parametrize("name,args,kwargs", FILTER_INVARIANT_CALL_PARAMS)
def test_invariant_no_numpy_filter(name, args, kwargs):
    """can_has returns False for NumPy array inputs (filters: image or hist)."""
    if "hist" in kwargs:
        assert not can_has(name, hist=kwargs["hist"])
    else:
        image = np.array([[0.0, 0.5], [0.2, 0.8]], dtype=np.float64)
        assert not can_has(name, image, *args, **kwargs)


# ---- Invariant 3: Shape/ndim of returned value matches scikit-image ----
@pytest.mark.cupy
@pytest.mark.parametrize("name,args,kwargs", INVARIANT_CALL_PARAMS)
def test_invariant_shape_match_scalar_result(name, args, kwargs, cupy, minimal_cupy_arrays_2d):
    """Backend return shape/ndim matches scikit-image for metrics (scalar/0-dim)."""
    a, b = minimal_cupy_arrays_2d
    np_a = np.asarray(cupy.asnumpy(a))
    np_b = np.asarray(cupy.asnumpy(b))
    ref = _skimage_reference_result(name, (np_a, np_b), args, kwargs)
    expected_shape, expected_ndim = _expected_shape_and_ndim(ref)

    impl = get_implementation(name)
    result = impl(a, b, *args, **kwargs)
    assert isinstance(result, cupy.ndarray)
    assert result.shape == expected_shape
    assert result.ndim == expected_ndim


@pytest.mark.cupy
@pytest.mark.parametrize("name,args,kwargs", TRANSFORM_INVARIANT_CALL_PARAMS)
def test_invariant_shape_match_array_result(name, args, kwargs, cupy, minimal_cupy_arrays_2d):
    """Backend return shape/ndim matches scikit-image for array output."""
    a, _ = minimal_cupy_arrays_2d
    np_a = np.asarray(cupy.asnumpy(a))
    ref = _skimage_reference_result(name, (np_a,), args, kwargs)
    expected_shape, expected_ndim = _expected_shape_and_ndim(ref)

    impl = get_implementation(name)
    if name == "skimage.transform:integrate":
        impl_ii = get_implementation("skimage.transform:integral_image")
        a = impl_ii(a)
    result = impl(a, *args, **kwargs)
    assert isinstance(result, cupy.ndarray)
    assert result.shape == expected_shape
    assert result.ndim == expected_ndim


@pytest.mark.cupy
@pytest.mark.parametrize("name,args,kwargs", FILTER_INVARIANT_CALL_PARAMS)
def test_invariant_shape_match_filter_result(name, args, kwargs, cupy, minimal_cupy_arrays_2d):
    """Backend return shape/ndim matches scikit-image for filter output."""
    impl = get_implementation(name)
    if "hist" in kwargs:
        np_hist = kwargs["hist"]
        cp_hist = make_hist_for_otsu(cupy)
        ref = _skimage_reference_result(name, (None,), (), {"hist": np_hist})
        expected_shape, expected_ndim = _expected_shape_and_ndim(ref)
        result = impl(None, hist=cp_hist)
    else:
        a, _ = minimal_cupy_arrays_2d
        np_a = np.asarray(cupy.asnumpy(a))
        ref = _skimage_reference_result(name, (np_a,), args, kwargs)
        expected_shape, expected_ndim = _expected_shape_and_ndim(ref)
        result = impl(a, *args, **kwargs)
    assert isinstance(result, cupy.ndarray)
    assert result.shape == expected_shape
    assert result.ndim == expected_ndim
