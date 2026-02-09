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
    (
        "skimage.filters:threshold_otsu",
        (),
        {"hist": "build_in_test"},
        None,
    ),  # scalar, hist provided
    ("skimage.filters:threshold_li", (), {}, None),
    ("skimage.filters:threshold_yen", (), {}, None),
    ("skimage.filters:threshold_yen", (), {"hist": "build_in_test"}, None),
    ("skimage.filters:threshold_isodata", (), {}, None),
    ("skimage.filters:threshold_isodata", (), {"hist": "build_in_test"}, None),
    ("skimage.filters:difference_of_gaussians", (2, 10), {}, (7, 7)),
    ("skimage.filters:prewitt", (), {}, (7, 7)),
    ("skimage.filters:scharr", (), {}, (7, 7)),
    ("skimage.filters:median", (), {}, (7, 7)),
    ("skimage.filters:laplace", (), {}, (7, 7)),  # CuCIM supports only ksize=3
    ("skimage.filters:roberts", (), {}, (7, 7)),
    ("skimage.filters:unsharp_mask", (), {}, (7, 7)),
]

# (name, args, kwargs, expected_shape) for segmentation (label image in, same shape out).
# join_segmentations: (s1, s2) -> array or (array, map, map); test uses two label images.
SEGMENTATION_CALL_PARAMS = [
    ("skimage.segmentation:clear_border", (), {}, (7, 7)),
    ("skimage.segmentation:expand_labels", (), {}, (7, 7)),
    ("skimage.segmentation:find_boundaries", (), {}, (7, 7)),
    ("skimage.segmentation:join_segmentations", (), {}, (7, 7)),
    ("skimage.segmentation:relabel_sequential", (), {}, (7, 7)),
]

# (name, args, kwargs, expected_shape) for measure. label: expected_shape (7,7) or None when return_num=True (then result is tuple).
MEASURE_CALL_PARAMS = [
    ("skimage.measure:label", (), {}, (7, 7)),
    ("skimage.measure:label", (), {"return_num": True}, None),  # returns (array, int)
]

# (name, args, kwargs, expected_shape) for morphology functions (all return same shape as input (7,7)).
MORPHOLOGY_CALL_PARAMS = [
    ("skimage.morphology:erosion", (), {}, (7, 7)),
    ("skimage.morphology:dilation", (), {}, (7, 7)),
    ("skimage.morphology:opening", (), {}, (7, 7)),
    ("skimage.morphology:closing", (), {}, (7, 7)),
    ("skimage.morphology:remove_small_objects", (), {}, (7, 7)),
    ("skimage.morphology:remove_small_holes", (), {}, (7, 7)),
    ("skimage.morphology:white_tophat", (), {}, (7, 7)),
    ("skimage.morphology:black_tophat", (), {}, (7, 7)),
]

# (name, args, kwargs, expected_shape) for exposure functions (all return same shape as image (7,7)).
EXPOSURE_CALL_PARAMS = [
    ("skimage.exposure:equalize_hist", (), {}, (7, 7)),
    ("skimage.exposure:equalize_adapthist", (), {}, (7, 7)),
    ("skimage.exposure:match_histograms", (), {}, (7, 7)),  # reference built in test
    ("skimage.exposure:rescale_intensity", (), {}, (7, 7)),
    ("skimage.exposure:adjust_gamma", (), {}, (7, 7)),
]

# (name, args, kwargs, expected_shape) for feature functions.
# peak_local_max expected_shape is None (variable: (n_peaks, 2)); test only checks ndim and shape[1].
# match_template: test builds 3x3 template for 7x7 image -> output (5, 5).
FEATURE_CALL_PARAMS = [
    ("skimage.feature:canny", (), {}, (7, 7)),
    ("skimage.feature:peak_local_max", (), {}, None),  # (n_peaks, 2)
    ("skimage.feature:match_template", (), {}, (5, 5)),
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
def test_transform_returns_cupy_array_with_shape(
    name, args, kwargs, expected_shape, cupy, require_cuda
):
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
def test_filter_returns_cupy_array(
    name, args, kwargs, expected_shape, cupy, require_cuda
):
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


@pytest.mark.cupy
@pytest.mark.parametrize("name,args,kwargs,expected_shape", SEGMENTATION_CALL_PARAMS)
def test_segmentation_returns_cupy_array(
    name, args, kwargs, expected_shape, cupy, require_cuda
):
    """Backend segmentation with CuPy input returns CuPy ndarray with expected shape."""
    impl = get_implementation(name)
    rng = np.random.default_rng(42)
    label_im = cupy.array((rng.random((7, 7)) * 3).astype(np.int32))
    if "join_segmentations" in name:
        label_im2 = cupy.array((rng.random((7, 7)) * 2).astype(np.int32))
        result = impl(label_im, label_im2, *args, **kwargs)
    else:
        result = impl(label_im, *args, **kwargs)
    if "relabel_sequential" in name:
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert isinstance(result[0], cupy.ndarray)
        assert result[0].shape == expected_shape
    else:
        assert isinstance(result, cupy.ndarray)
        assert result.shape == expected_shape


@pytest.mark.cupy
@pytest.mark.parametrize("name,args,kwargs,expected_shape", MEASURE_CALL_PARAMS)
def test_measure_returns_cupy_array(
    name, args, kwargs, expected_shape, cupy, require_cuda
):
    """Backend measure with CuPy input returns CuPy ndarray (or (array, int) for label(return_num=True))."""
    impl = get_implementation(name)
    rng = np.random.default_rng(42)
    # Binary-like image for label
    image = cupy.array((rng.random((7, 7)) > 0.5).astype(np.int32))
    result = impl(image, *args, **kwargs)
    if kwargs.get("return_num"):
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], cupy.ndarray)
        assert result[0].shape == (7, 7)
        assert isinstance(result[1], int)
    else:
        assert isinstance(result, cupy.ndarray)
        assert result.shape == expected_shape


@pytest.mark.cupy
@pytest.mark.parametrize("name,args,kwargs,expected_shape", MORPHOLOGY_CALL_PARAMS)
def test_morphology_returns_cupy_array(
    name, args, kwargs, expected_shape, cupy, require_cuda
):
    """Backend morphology with CuPy input returns CuPy ndarray with expected shape."""
    impl = get_implementation(name)
    rng = np.random.default_rng(42)
    # Boolean for erosion/dilation/opening/closing/remove_small_*; float for white_tophat/black_tophat
    gray_float_only = ("white_tophat", "black_tophat")
    if any(g in name for g in gray_float_only):
        image = cupy.array(rng.random((7, 7), dtype=np.float64))
    else:
        image = cupy.array((rng.random((7, 7)) > 0.5), dtype=bool)
    result = impl(image, *args, **kwargs)
    assert isinstance(result, cupy.ndarray)
    assert result.shape == expected_shape


@pytest.mark.cupy
@pytest.mark.parametrize("name,args,kwargs,expected_shape", EXPOSURE_CALL_PARAMS)
def test_exposure_returns_cupy_array(
    name, args, kwargs, expected_shape, cupy, require_cuda
):
    """Backend exposure with CuPy input returns CuPy ndarray with expected shape."""
    impl = get_implementation(name)
    rng = np.random.default_rng(42)
    image = cupy.array(rng.random((7, 7), dtype=np.float64))
    if name == "skimage.exposure:match_histograms":
        reference = cupy.array(rng.random((7, 7), dtype=np.float64))
        result = impl(image, reference, *args, **kwargs)
    else:
        result = impl(image, *args, **kwargs)
    assert isinstance(result, cupy.ndarray)
    assert result.shape == expected_shape


@pytest.mark.cupy
@pytest.mark.parametrize("name,args,kwargs,expected_shape", FEATURE_CALL_PARAMS)
def test_feature_returns_cupy_array(
    name, args, kwargs, expected_shape, cupy, require_cuda
):
    """Backend feature with CuPy input returns CuPy ndarray with expected shape."""
    impl = get_implementation(name)
    rng = np.random.default_rng(42)
    image = cupy.array(rng.random((7, 7), dtype=np.float64))
    if name == "skimage.feature:match_template":
        template = cupy.array(rng.random((3, 3), dtype=np.float64))
        result = impl(image, template, *args, **kwargs)
    else:
        result = impl(image, *args, **kwargs)
    assert isinstance(result, cupy.ndarray)
    if expected_shape is None:
        # peak_local_max: (n_peaks, 2)
        assert result.ndim == 2
        assert result.shape[1] == 2
    else:
        assert result.shape == expected_shape


def test_get_implementation_unsupported_raises():
    """get_implementation for unsupported name raises LookupError."""
    with pytest.raises(LookupError, match="Unsupported"):
        get_implementation("skimage.metrics:nonexistent")


@pytest.mark.cupy
@pytest.mark.parametrize(
    "name,call_kwargs,match",
    [
        (
            "skimage.filters:laplace",
            {"ksize": 5},
            "supports laplace\\(ksize=3\\) only",
        ),
        (
            "skimage.filters:median",
            {"behavior": "rank"},
            "does not support median\\(behavior='rank'\\)",
        ),
    ],
)
def test_implementation_raises_for_unsupported_params(
    name, call_kwargs, match, cupy, require_cuda
):
    """Calling backend implementation with CuCIM-unsupported params raises ValueError."""
    impl = get_implementation(name)
    rng = np.random.default_rng(42)
    image = cupy.array(rng.random((7, 7), dtype=np.float64))
    with pytest.raises(ValueError, match=match):
        impl(image, **call_kwargs)
