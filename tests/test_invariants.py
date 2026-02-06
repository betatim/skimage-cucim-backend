"""Per-function invariant tests: CuPy in->out, no NumPy, shape match."""

import numpy as np
import pytest

from skimage_cucim_backend.implementations import can_has, get_implementation
from skimage_cucim_backend.information import SUPPORTED_FUNCTIONS

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
    # structural_similarity: default full=False -> scalar; data_range optional
    ("skimage.metrics:structural_similarity", (), {}),
    ("skimage.metrics:structural_similarity", (), {"data_range": 1.0}),
]

def test_all_supported_functions_covered_in_invariant_call_params():
    """Every SUPPORTED_FUNCTIONS entry appears at least once in INVARIANT_CALL_PARAMS."""
    param_names = {name for name, _, _ in INVARIANT_CALL_PARAMS}
    for fn in SUPPORTED_FUNCTIONS:
        assert fn in param_names, f"{fn} missing from INVARIANT_CALL_PARAMS"


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
    """can_has returns False for NumPy array inputs."""
    a = np.array([[0.0, 0.5], [0.2, 0.8]], dtype=np.float64)
    b = np.array([[0.1, 0.4], [0.3, 0.9]], dtype=np.float64)
    assert can_has(name, a, b, *args, **kwargs) is False


# ---- Invariant 3: Shape of returned array matches scikit-image ----
# For scalar-returning metrics we only run the structure; shape N/A.
@pytest.mark.cupy
@pytest.mark.parametrize("name,args,kwargs", INVARIANT_CALL_PARAMS)
def test_invariant_shape_match_scalar_result(name, args, kwargs, cupy, minimal_cupy_arrays_2d):
    """For scalar-returning metrics, backend returns 0-dim CuPy array; shape N/A."""
    a, b = minimal_cupy_arrays_2d
    impl = get_implementation(name)
    result = impl(a, b, *args, **kwargs)
    assert isinstance(result, cupy.ndarray) and result.ndim == 0
