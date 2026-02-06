"""Shared test data and fixtures. CuPy/cucim tests skip when unavailable."""

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "cupy: mark test as requiring CuPy (skip if not installed)"
    )
    config.addinivalue_line(
        "markers",
        "cuda: mark test as requiring a working CUDA runtime (skip if CuPy has no GPU)",
    )


def _cuda_available():
    """Return True if CuPy is importable and can run a minimal GPU op (CUDA available)."""
    try:
        import cupy as cp
        x = cp.array([1.0])
        _ = x + 1  # triggers kernel
        return True
    except (ImportError, RuntimeError):
        return False


@pytest.fixture
def cupy():
    """CuPy module; skip test if not available."""
    return pytest.importorskip("cupy")


@pytest.fixture
def require_cuda():
    """Skip test if CuPy cannot run a minimal GPU op (no CUDA runtime)."""
    if not _cuda_available():
        pytest.skip("CUDA runtime not available (CuPy installed but no GPU/CUDA_PATH)")


@pytest.fixture
def minimal_cupy_arrays_2d(cupy, require_cuda):
    """Minimal 2D CuPy arrays for metrics (same shape). Skip if CUDA runtime unavailable."""
    return cupy.array([[0.0, 0.5], [0.2, 0.8]], dtype=cupy.float64), cupy.array(
        [[0.1, 0.4], [0.3, 0.9]], dtype=cupy.float64
    )
