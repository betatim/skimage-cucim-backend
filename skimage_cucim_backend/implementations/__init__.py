"""Backend namespace: can_has and get_implementation."""

import importlib
import sys

from skimage_cucim_backend.information import SUPPORTED_FUNCTIONS


def _is_cupy_array(obj):
    """Return True if obj is a CuPy array (or array-like)."""
    if obj is None:
        return False

    cupy = sys.modules.get("cupy")
    if cupy is None:
        return False
    return isinstance(obj, cupy.ndarray)


def _looks_like_array(obj):
    """True if obj has array-like attributes (ndim, shape). NumPy and CuPy arrays have both."""
    if obj is None:
        return False
    return hasattr(obj, "ndim") and hasattr(obj, "shape")


def _first_array_from_args(args, kwargs):
    """Return the first argument that looks like an array (has ndim and shape).

    Looks inside tuples/lists (e.g. hist=(counts, bin_centers) for threshold_otsu).
    """

    def first_array_in(obj):
        if obj is None:
            return None
        if _looks_like_array(obj):
            return obj
        if isinstance(obj, (tuple, list)):
            for item in obj:
                found = first_array_in(item)
                if found is not None:
                    return found
        return None

    for obj in args:
        found = first_array_in(obj)
        if found is not None:
            return found
    for obj in kwargs.values():
        found = first_array_in(obj)
        if found is not None:
            return found
    return None


def _reject_dispatch(name, args, kwargs):
    """Return True when the backend cannot handle this call due to unsupported parameters.

    CuCIM has limitations that scikit-image does not; for those cases we reject
    dispatching so scikit-image will use its own implementation.
    """
    if name == "skimage.filters:laplace":
        if kwargs.get("ksize", 3) != 3:
            return True  # CuCIM supports only ksize=3
    elif name == "skimage.filters:median":
        if kwargs.get("behavior", "ndimage") == "rank":
            return True  # CuCIM does not implement rank behavior
    return False


def can_has(name, *args, **kwargs):
    """Return True only when the backend can handle this call (CuPy array inputs only)."""
    if name not in SUPPORTED_FUNCTIONS:
        return False
    if _reject_dispatch(name, args, kwargs):
        return False
    arr = _first_array_from_args(args, kwargs)
    # XXX we need to handle the case if there are no array inputs
    # XXX for now we retur nFalse but really we should dispatch
    # XXX to cucim for these functions. Something for later.
    if arr is None:
        return False
    return _is_cupy_array(arr)


def get_implementation(name):
    """Return the backend implementation callable for the given dispatch name."""
    if name not in SUPPORTED_FUNCTIONS:
        raise LookupError(f"Unsupported function: {name}")
    # Parse "skimage.metrics:mean_squared_error" -> module path + func name
    _, rest = name.split(".", maxsplit=1)
    module_path, func_name = rest.rsplit(":", maxsplit=1)

    if module_path in ("metrics", "transform", "filters", "feature", "exposure", "morphology"):
        mod = importlib.import_module(
            f"skimage_cucim_backend.implementations.{module_path}"
        )
    else:
        raise LookupError(f"No implementation for module path: {module_path}")

    func = getattr(mod, func_name, None)
    if func is None:
        raise LookupError(f"No implementation for: {name}")
    return func
