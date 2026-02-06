"""Test get_implementation returns callables that work with CuPy inputs."""

import pytest

from skimage_cucim_backend.implementations import get_implementation
from skimage_cucim_backend.information import SUPPORTED_FUNCTIONS

# (name, args, kwargs) for each metric call to test. Arrays a, b are passed in the test.
METRIC_CALL_PARAMS = [
    ("skimage.metrics:mean_squared_error", (), {}),
    ("skimage.metrics:normalized_root_mse", (), {}),
    ("skimage.metrics:normalized_root_mse", (), {"normalization": "euclidean"}),
    ("skimage.metrics:peak_signal_noise_ratio", (), {}),
    ("skimage.metrics:peak_signal_noise_ratio", (), {"data_range": 1.0}),
    ("skimage.metrics:structural_similarity", (), {}),
    ("skimage.metrics:structural_similarity", (), {"data_range": 1.0}),
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
    a = cupy.array([[0.0, 0.5], [0.2, 0.8]], dtype=cupy.float64)
    b = cupy.array([[0.1, 0.4], [0.3, 0.9]], dtype=cupy.float64)
    result = impl(a, b, *args, **kwargs)
    assert isinstance(result, cupy.ndarray) and result.ndim == 0


def test_get_implementation_unsupported_raises():
    """get_implementation for unsupported name raises LookupError."""
    with pytest.raises(LookupError, match="Unsupported"):
        get_implementation("skimage.metrics:nonexistent")
