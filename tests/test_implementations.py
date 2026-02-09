"""Unit tests for implementations module: helpers and edge cases for can_has / get_implementation."""

import importlib
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from skimage_cucim_backend._testing import make_hist_for_otsu
import skimage_cucim_backend.implementations as impl_mod
from skimage_cucim_backend.implementations import (
    _first_array_from_args,
    _is_cupy_array,
    _looks_like_array,
    can_has,
    get_implementation,
)
from skimage_cucim_backend.information import SUPPORTED_FUNCTIONS


# ---------------------------------------------------------------------------
# _looks_like_array
# ---------------------------------------------------------------------------


def test_looks_like_array_none():
    assert not _looks_like_array(None)


def test_looks_like_array_numpy():
    assert _looks_like_array(np.array([[1, 2], [3, 4]]))


def test_looks_like_array_plain_objects():
    assert not _looks_like_array(object())
    assert not _looks_like_array("hello")
    assert not _looks_like_array(42)
    assert not _looks_like_array([])


def test_looks_like_array_only_ndim():
    """Object with ndim but no shape is not array-like."""
    class OnlyNdim:
        ndim = 2
    assert not _looks_like_array(OnlyNdim())


def test_looks_like_array_only_shape():
    """Object with shape but no ndim is not array-like."""
    class OnlyShape:
        shape = (3, 4)
    assert not _looks_like_array(OnlyShape())


def test_looks_like_array_mock_array_like():
    """Object with both ndim and shape is array-like."""
    class ArrayLike:
        ndim = 2
        shape = (3, 4)
    assert _looks_like_array(ArrayLike())


@pytest.mark.cupy
def test_looks_like_array_cupy(cupy):
    assert _looks_like_array(cupy.array([1, 2, 3]))


# ---------------------------------------------------------------------------
# _first_array_from_args
# ---------------------------------------------------------------------------


def test_first_array_from_args_empty():
    assert _first_array_from_args((), {}) is None


def test_first_array_from_args_array_in_first_arg():
    a = np.array([1])
    assert _first_array_from_args((a, 2, 3), {}) is a


def test_first_array_from_args_array_in_second_arg():
    a = np.array([1])
    assert _first_array_from_args((1, a, 3), {}) is a


def test_first_array_from_args_array_in_kwargs():
    a = np.array([1])
    assert _first_array_from_args((), {"x": a}) is a


def test_first_array_from_args_args_before_kwargs():
    a1 = np.array([1])
    a2 = np.array([2])
    assert _first_array_from_args((a1,), {"x": a2}) is a1


def test_first_array_from_args_array_inside_tuple_in_args():
    a = np.array([1])
    assert _first_array_from_args(((a, 2),), {}) is a


def test_first_array_from_args_array_inside_list_in_args():
    a = np.array([1])
    assert _first_array_from_args(([a],), {}) is a


def test_first_array_from_args_nested_tuple_list():
    a = np.array([1])
    assert _first_array_from_args((([a],),), {}) is a


def test_first_array_from_args_threshold_otsu_style_hist():
    counts = np.array([1.0, 2.0, 3.0])
    bin_centers = np.array([0.0, 0.5, 1.0])
    result = _first_array_from_args((), {"hist": (counts, bin_centers)})
    assert result is counts


def test_first_array_from_args_array_inside_list_in_kwargs():
    a = np.array([1])
    assert _first_array_from_args((), {"hist": [a, np.array([2])]}) is a


def test_first_array_from_args_no_array_in_nested():
    assert _first_array_from_args(((1, 2), [3, 4]), {"a": 1}) is None


def test_first_array_from_args_none_only():
    assert _first_array_from_args((None, None), {"x": None}) is None


# ---------------------------------------------------------------------------
# _is_cupy_array
# ---------------------------------------------------------------------------


def test_is_cupy_array_none():
    assert not _is_cupy_array(None)


def test_is_cupy_array_numpy():
    assert not _is_cupy_array(np.array([1, 2, 3]))


def test_is_cupy_array_plain_objects():
    assert not _is_cupy_array(object())
    assert not _is_cupy_array("hello")
    assert not _is_cupy_array(42)


@pytest.mark.cupy
def test_is_cupy_array_cupy(cupy):
    assert _is_cupy_array(cupy.array([1, 2, 3]))


def test_is_cupy_array_when_cupy_not_in_sys_modules():
    """When cupy is not in sys.modules, _is_cupy_array returns False for any object."""
    cupy_ref = sys.modules.get("cupy")
    try:
        if "cupy" in sys.modules:
            del sys.modules["cupy"]
        # _is_cupy_array reads sys.modules.get("cupy") on each call, so no reload needed
        assert not _is_cupy_array(None)
        assert not _is_cupy_array(np.array([1]))
    finally:
        if cupy_ref is not None:
            sys.modules["cupy"] = cupy_ref


# ---------------------------------------------------------------------------
# can_has
# ---------------------------------------------------------------------------


def test_can_has_unsupported_name():
    cupy = pytest.importorskip("cupy")
    a = cupy.array([1.0])
    assert not can_has("skimage.metrics:unknown", a, a)


def test_can_has_supported_name_no_array():
    """No array in args/kwargs -> False (current contract)."""
    assert not can_has("skimage.metrics:mean_squared_error")
    assert not can_has("skimage.metrics:mean_squared_error", 1, 2)


def test_can_has_supported_name_numpy_array():
    a = np.array([1.0])
    b = np.array([0.5])
    assert not can_has("skimage.metrics:mean_squared_error", a, b)


@pytest.mark.cupy
def test_can_has_supported_name_cupy_array(cupy):
    a = cupy.array([1.0])
    b = cupy.array([0.5])
    assert can_has("skimage.metrics:mean_squared_error", a, b)


@pytest.mark.cupy
def test_can_has_first_array_from_nested_hist_cupy(cupy):
    hist = make_hist_for_otsu(cupy)
    assert can_has("skimage.filters:threshold_otsu", hist=hist)


def test_can_has_first_array_from_nested_hist_numpy():
    hist = make_hist_for_otsu(np)
    assert not can_has("skimage.filters:threshold_otsu", hist=hist)


# ---------------------------------------------------------------------------
# get_implementation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", SUPPORTED_FUNCTIONS)
def test_get_implementation_supported_returns_callable(name):
    impl = get_implementation(name)
    assert callable(impl)


def test_get_implementation_unsupported_raises():
    with pytest.raises(LookupError, match="Unsupported"):
        get_implementation("skimage.metrics:nonexistent")


def test_get_implementation_unknown_module_path_raises():
    """When SUPPORTED_FUNCTIONS includes a name with unknown module path, raises LookupError."""
    name = "skimage.other:foo"
    with patch.object(impl_mod, "SUPPORTED_FUNCTIONS", list(SUPPORTED_FUNCTIONS) + [name]):
        with pytest.raises(LookupError, match="No implementation for module path"):
            get_implementation(name)


def test_get_implementation_missing_function_in_module_raises():
    """When the resolved module does not define the function, raises LookupError."""
    # Patch import_module to return a mock module where getattr(..., func_name, None) is None
    def fake_import_module(path):
        if path == "skimage_cucim_backend.implementations.metrics":
            mod = MagicMock()
            mod.mean_squared_error = None  # getattr(mod, "mean_squared_error", None) -> None
            return mod
        return importlib.import_module(path)

    with patch("skimage_cucim_backend.implementations.importlib.import_module", side_effect=fake_import_module):
        with pytest.raises(LookupError, match="No implementation for:"):
            get_implementation("skimage.metrics:mean_squared_error")
