"""Test get_implementation returns callables that work with CuPy inputs."""

import pytest

import numpy as np

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

# (name, args, kwargs, expected_shape) for transform functions that return arrays.
# rescale(7x7 image, 0.5) -> (4, 4)
TRANSFORM_CALL_PARAMS = [
    ("skimage.transform:resize", ((10, 10),), {}, (10, 10)),
    ("skimage.transform:rescale", (0.5,), {}, (4, 4)),
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
    result = impl(image, *args, **kwargs)
    assert isinstance(result, cupy.ndarray)
    assert result.shape == expected_shape


def test_get_implementation_unsupported_raises():
    """get_implementation for unsupported name raises LookupError."""
    with pytest.raises(LookupError, match="Unsupported"):
        get_implementation("skimage.metrics:nonexistent")
