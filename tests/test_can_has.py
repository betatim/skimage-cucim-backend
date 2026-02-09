"""Test can_has: CuPy -> True, NumPy/unsupported -> False."""

import numpy as np
import pytest

from skimage_cucim_backend._testing import identity_map, make_hist_for_otsu
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
    ("skimage.transform:warp", True, True, {"output_shape": (7, 7)}, (identity_map,)),
    ("skimage.transform:warp", False, False, {"output_shape": (7, 7)}, (identity_map,)),
    ("skimage.transform:resize_local_mean", True, True, {}, ((4, 4),)),
    ("skimage.transform:resize_local_mean", False, False, {}, ((4, 4),)),
    ("skimage.transform:downscale_local_mean", True, True, {}, ((2, 2),)),
    ("skimage.transform:downscale_local_mean", False, False, {}, ((2, 2),)),
    ("skimage.transform:integral_image", True, True, {}, ()),
    ("skimage.transform:integral_image", False, False, {}, ()),
    ("skimage.transform:integrate", True, True, {}, ([(0, 0)], [(1, 1)])),
    ("skimage.transform:integrate", False, False, {}, ([(0, 0)], [(1, 1)])),
    ("skimage.transform:pyramid_reduce", True, True, {}, ()),
    ("skimage.transform:pyramid_reduce", False, False, {}, ()),
    ("skimage.transform:pyramid_expand", True, True, {}, ()),
    ("skimage.transform:pyramid_expand", False, False, {}, ()),
    ("skimage.transform:swirl", True, True, {}, ()),
    ("skimage.transform:swirl", False, False, {}, ()),
    ("skimage.transform:warp_polar", True, True, {"output_shape": (7, 7)}, ()),
    ("skimage.transform:warp_polar", False, False, {"output_shape": (7, 7)}, ()),
    ("skimage.filters:gaussian", True, True, {}, ()),
    ("skimage.filters:gaussian", False, False, {}, ()),
    ("skimage.filters:sobel", True, True, {}, ()),
    ("skimage.filters:sobel", False, False, {}, ()),
    ("skimage.filters:threshold_otsu", True, True, {}, ()),
    ("skimage.filters:threshold_otsu", False, False, {}, ()),
    ("skimage.filters:threshold_li", True, True, {}, ()),
    ("skimage.filters:threshold_li", False, False, {}, ()),
    ("skimage.filters:threshold_yen", True, True, {}, ()),
    ("skimage.filters:threshold_yen", False, False, {}, ()),
    ("skimage.filters:threshold_isodata", True, True, {}, ()),
    ("skimage.filters:threshold_isodata", False, False, {}, ()),
    ("skimage.filters:difference_of_gaussians", True, True, {}, (2, 10)),
    ("skimage.filters:difference_of_gaussians", False, False, {}, (2, 10)),
    ("skimage.filters:prewitt", True, True, {}, ()),
    ("skimage.filters:prewitt", False, False, {}, ()),
    ("skimage.filters:scharr", True, True, {}, ()),
    ("skimage.filters:scharr", False, False, {}, ()),
    ("skimage.filters:median", True, True, {}, ()),
    ("skimage.filters:median", False, False, {}, ()),
]


@pytest.mark.parametrize("use_cupy,expected", [(True, True), (False, False)])
def test_can_has_threshold_otsu_hist_only(use_cupy, expected):
    """can_has(threshold_otsu) with hist=(counts, bin_centers): True if CuPy, False if NumPy."""
    if use_cupy:
        cupy = pytest.importorskip("cupy")
        xp = cupy
    else:
        xp = np
    hist = make_hist_for_otsu(xp)
    result = can_has("skimage.filters:threshold_otsu", hist=hist)
    assert result is expected


@pytest.mark.parametrize("name", ["skimage.filters:threshold_yen", "skimage.filters:threshold_isodata"])
@pytest.mark.parametrize("use_cupy,expected", [(True, True), (False, False)])
def test_can_has_threshold_hist_only(name, use_cupy, expected):
    """can_has(threshold_yen/isodata) with hist=(counts, bin_centers): True if CuPy, False if NumPy."""
    if use_cupy:
        cupy = pytest.importorskip("cupy")
        xp = cupy
    else:
        xp = np
    hist = make_hist_for_otsu(xp)
    result = can_has(name, hist=hist)
    assert result is expected


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
