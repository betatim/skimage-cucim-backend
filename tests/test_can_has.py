"""Test can_has: CuPy -> True, NumPy/unsupported -> False."""

import numpy as np
import pytest

from skimage_cucim_backend.implementations import can_has
from skimage_cucim_backend.information import SUPPORTED_FUNCTIONS

# (name, use_cupy, expected, kwargs, extra_args) for each can_has call.
# extra_args=None means pass (a, b) for two-array metrics; otherwise pass (a, *extra_args).
CAN_HAS_PARAMS = [
    ("skimage.metrics:mean_squared_error", True, True, {}, None),
    ("skimage.metrics:mean_squared_error", False, False, {}, None),
    ("skimage.metrics:normalized_root_mse", True, True, {}, None),
    ("skimage.metrics:normalized_root_mse", False, False, {}, None),
    ("skimage.metrics:normalized_root_mse", True, True, {"normalization": "euclidean"}, None),
    ("skimage.metrics:peak_signal_noise_ratio", True, True, {}, None),
    ("skimage.metrics:peak_signal_noise_ratio", False, False, {}, None),
    ("skimage.metrics:peak_signal_noise_ratio", True, True, {"data_range": 1.0}, None),
    ("skimage.metrics:structural_similarity", True, True, {"data_range": 1.0}, None),
    ("skimage.metrics:structural_similarity", False, False, {}, None),
    ("skimage.metrics:normalized_mutual_information", True, True, {}, None),
    ("skimage.metrics:normalized_mutual_information", False, False, {}, None),
    ("skimage.metrics:normalized_mutual_information", True, True, {"bins": 10}, None),
    ("skimage.metrics:unknown", True, False, {}, None),
    ("skimage.metrics:unknown", False, False, {}, None),
    ("skimage.transform:resize", True, True, {}, ((10, 10),)),
    ("skimage.transform:resize", False, False, {}, ((10, 10),)),
    ("skimage.transform:rescale", True, True, {}, (0.5,)),
    ("skimage.transform:rescale", False, False, {}, (0.5,)),
    ("skimage.transform:rotate", True, True, {}, (45,)),
    ("skimage.transform:rotate", False, False, {}, (45,)),
]


@pytest.mark.parametrize("name,use_cupy,expected,kwargs,extra_args", CAN_HAS_PARAMS)
def test_can_has(name, use_cupy, expected, kwargs, extra_args):
    """can_has returns True only for supported names with CuPy array inputs."""
    if use_cupy:
        cupy = pytest.importorskip("cupy")
        a = cupy.array([0.0, 1.0])
        b = cupy.array([0.0, 0.5])
    else:
        a = np.array([0.0, 1.0])
        b = np.array([0.0, 0.5])

    if extra_args is None:
        result = can_has(name, a, b, **kwargs)
    else:
        result = can_has(name, a, *extra_args, **kwargs)
    assert result is expected
