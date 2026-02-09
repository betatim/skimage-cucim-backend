"""Test get_implementation returns callables that work with CuPy inputs."""

import pytest

import numpy as np

from skimage_cucim_backend._testing import identity_map, make_hist_for_otsu
from skimage_cucim_backend.implementations import get_implementation
from skimage_cucim_backend.information import SUPPORTED_FUNCTIONS

# (name, args, kwargs) for each metric call to test. Arrays a, b are passed in the test.
METRIC_CALL_PARAMS = [
    ("skimage.metrics:mean_squared_error", (), {}),
    ("skimage.metrics:normalized_root_mse", (), {}),
    ("skimage.metrics:normalized_root_mse", (), {"normalization": "euclidean"}),
    ("skimage.metrics:peak_signal_noise_ratio", (), {}),
    ("skimage.metrics:peak_signal_noise_ratio", (), {"data_range": 1.0}),
    # structural_similarity: pass data_range explicitly (recommended for float images in both skimage and CuCIM)
    ("skimage.metrics:structural_similarity", (), {"data_range": 1.0}),
    ("skimage.metrics:structural_similarity", (), {"data_range": 1.0, "win_size": 7}),
    ("skimage.metrics:normalized_mutual_information", (), {}),
    ("skimage.metrics:normalized_mutual_information", (), {"bins": 10}),
]

# (name, args, kwargs, expected_shape) for filter functions. expected_shape=None means scalar (0-dim).
# For threshold_otsu: first row = image provided; second row = hist provided (no image).
FILTERS_CALL_PARAMS = [
    ("skimage.filters:gaussian", (), {"sigma": 1.0}, (7, 7)),
    ("skimage.filters:sobel", (), {}, (7, 7)),
    ("skimage.filters:threshold_otsu", (), {}, None),  # scalar, image provided
    ("skimage.filters:threshold_otsu", (), {"hist": "build_in_test"}, None),  # scalar, hist provided
    ("skimage.filters:difference_of_gaussians", (2, 10), {}, (7, 7)),
    ("skimage.filters:prewitt", (), {}, (7, 7)),
    ("skimage.filters:scharr", (), {}, (7, 7)),
    ("skimage.filters:median", (), {}, (7, 7)),
]

# (name, args, kwargs, expected_shape) for transform functions that return arrays.
# rescale(7x7 image, 0.5) -> (4, 4)
TRANSFORM_CALL_PARAMS = [
    ("skimage.transform:resize", ((10, 10),), {}, (10, 10)),
    ("skimage.transform:rescale", (0.5,), {}, (4, 4)),
    ("skimage.transform:rotate", (45,), {}, (7, 7)),
    ("skimage.transform:warp", (identity_map,), {"output_shape": (7, 7)}, (7, 7)),
    ("skimage.transform:resize_local_mean", ((4, 4),), {}, (4, 4)),
    ("skimage.transform:downscale_local_mean", ((2, 2),), {}, (4, 4)),
    ("skimage.transform:integral_image", (), {}, (7, 7)),
    ("skimage.transform:integrate", ([(0, 0)], [(1, 1)]), {}, (1,)),
    ("skimage.transform:pyramid_reduce", (), {}, (4, 4)),
    ("skimage.transform:pyramid_expand", (), {}, (14, 14)),
    ("skimage.transform:swirl", (), {}, (7, 7)),
    ("skimage.transform:warp_polar", (), {"output_shape": (7, 7)}, (7, 7)),
]


@pytest.mark.parametrize("name", SUPPORTED_FUNCTIONS)
def test_get_implementation_returns_callable(name):
    """get_implementation(name) returns a callable."""
    impl = get_implementation(name)
    assert callable(impl)


@pytest.mark.cupy
@pytest.mark.parametrize("name,args,kwargs", METRIC_CALL_PARAMS)
def test_metric_returns_0dim_cupy_array(name, args, kwargs, cupy, require_cuda):
    """Backend metrics with CuPy inputs return a 0-dim CuPy array."""
    impl = get_implementation(name)
    rng = np.random.default_rng(42)
    a = cupy.array(rng.random((7, 7), dtype=np.float64))
    b = cupy.array(rng.random((7, 7), dtype=np.float64))
    result = impl(a, b, *args, **kwargs)
    assert isinstance(result, cupy.ndarray) and result.ndim == 0


@pytest.mark.cupy
@pytest.mark.parametrize("name,args,kwargs,expected_shape", TRANSFORM_CALL_PARAMS)
def test_transform_returns_cupy_array_with_shape(name, args, kwargs, expected_shape, cupy, require_cuda):
    """Backend transform with CuPy input returns a CuPy ndarray with the expected shape."""
    impl = get_implementation(name)
    rng = np.random.default_rng(42)
    image = cupy.array(rng.random((7, 7), dtype=np.float64))
    if name == "skimage.transform:integrate":
        impl_ii = get_implementation("skimage.transform:integral_image")
        image = impl_ii(image)
    result = impl(image, *args, **kwargs)
    assert isinstance(result, cupy.ndarray)
    assert result.shape == expected_shape


@pytest.mark.cupy
@pytest.mark.parametrize("name,args,kwargs,expected_shape", FILTERS_CALL_PARAMS)
def test_filter_returns_cupy_array(name, args, kwargs, expected_shape, cupy, require_cuda):
    """Backend filter with CuPy input returns CuPy ndarray; shape or 0-dim for scalar."""
    impl = get_implementation(name)
    rng = np.random.default_rng(42)
    # threshold_otsu(hist=...) case: no image, build hist with CuPy
    if kwargs.get("hist") == "build_in_test":
        kwargs = {"hist": make_hist_for_otsu(cupy)}
        result = impl(None, *args, **kwargs)
    else:
        image = cupy.array(rng.random((7, 7), dtype=np.float64))
        result = impl(image, *args, **kwargs)
    assert isinstance(result, cupy.ndarray)
    if expected_shape is None:
        assert result.ndim == 0
    else:
        assert result.shape == expected_shape


def test_get_implementation_unsupported_raises():
    """get_implementation for unsupported name raises LookupError."""
    with pytest.raises(LookupError, match="Unsupported"):
        get_implementation("skimage.metrics:nonexistent")
