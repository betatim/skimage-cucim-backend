"""Test helpers for parametrized tests (e.g. transform callables). Not part of public API."""


def identity_map(coords):
    """Identity inverse_map for warp: output coords -> input coords."""
    return coords
