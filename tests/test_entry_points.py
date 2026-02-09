"""Test that entry points are registered and expose the expected API."""

import pytest

from skimage_cucim_backend.information import SUPPORTED_FUNCTIONS


def test_backend_infos_entry_point_cucim():
    """Entry point skimage_backend_infos has name 'cucim' and returns BackendInformation."""
    eps = pytest.importorskip("importlib.metadata").entry_points(
        group="skimage_backend_infos"
    )
    cucim_info = None
    for ep in eps:
        if ep.name == "cucim":
            cucim_info = ep
            break
    assert cucim_info is not None, (
        "Entry point cucim not found in skimage_backend_infos"
    )
    info_fn = cucim_info.load()
    backend_info = info_fn()
    assert hasattr(backend_info, "supported_functions")
    assert set(backend_info.supported_functions) == set(SUPPORTED_FUNCTIONS)


def test_backends_entry_point_cucim():
    """Entry point skimage_backends has name 'cucim' and namespace has can_has and get_implementation."""
    eps = pytest.importorskip("importlib.metadata").entry_points(
        group="skimage_backends"
    )
    cucim_impl = None
    for ep in eps:
        if ep.name == "cucim":
            cucim_impl = ep
            break
    assert cucim_impl is not None, "Entry point cucim not found in skimage_backends"
    namespace = cucim_impl.load()
    assert callable(getattr(namespace, "can_has", None))
    assert callable(getattr(namespace, "get_implementation", None))
