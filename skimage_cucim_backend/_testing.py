"""Test helpers for parametrized tests (e.g. transform callables). Not part of public API."""


def identity_map(coords):
    """Identity inverse_map for warp: output coords -> input coords."""
    return coords


def make_hist_for_otsu(xp, n=10):
    """(counts, bin_centers) valid for threshold_otsu(hist=...). xp is np or cp."""
    counts = xp.arange(1, n + 1, dtype=xp.float64)
    bin_centers = xp.linspace(0.0, 1.0, n)
    return (counts, bin_centers)
