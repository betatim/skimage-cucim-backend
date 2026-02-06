"""Test can_has: CuPy -> True, NumPy/unsupported -> False."""

import numpy as np
import pytest

from skimage_cucim_backend.implementations import can_has
from skimage_cucim_backend.information import SUPPORTED_FUNCTIONS

# (name, use_cupy, expected, kwargs) for each can_has call to test. Arrays a, b are created in the test.
CAN_HAS_PARAMS = [
    ("skimage.metrics:mean_squared_error", True, True, {}),
    ("skimage.metrics:mean_squared_error", False, False, {}),
    ("skimage.metrics:normalized_root_mse", True, True, {}),
    ("skimage.metrics:normalized_root_mse", False, False, {}),
    ("skimage.metrics:normalized_root_mse", True, True, {"normalization": "euclidean"}),
    ("skimage.metrics:peak_signal_noise_ratio", True, True, {}),
    ("skimage.metrics:peak_signal_noise_ratio", False, False, {}),
    ("skimage.metrics:peak_signal_noise_ratio", True, True, {"data_range": 1.0}),
    ("skimage.metrics:structural_similarity", True, True, {}),
    ("skimage.metrics:structural_similarity", False, False, {}),
    ("skimage.metrics:structural_similarity", True, True, {"data_range": 1.0}),
    ("skimage.metrics:unknown", True, False, {}),
    ("skimage.metrics:unknown", False, False, {}),
]


@pytest.mark.parametrize("name,use_cupy,expected,kwargs", CAN_HAS_PARAMS)
def test_can_has(name, use_cupy, expected, kwargs):
    """can_has returns True only for supported names with CuPy array inputs."""
    if use_cupy:
        cupy = pytest.importorskip("cupy")
        a = cupy.array([0.0, 1.0])
        b = cupy.array([0.0, 0.5])
    else:
        a = np.array([0.0, 1.0])
        b = np.array([0.0, 0.5])

    result = can_has(name, a, b, **kwargs)
    assert result is expected
