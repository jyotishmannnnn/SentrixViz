"""RollingNormalizer (Phase 3): bounded memory, stable + gradual adaptation.

These tests exercise the policy directly (the live seam) with no rendering, so
they assert the numeric behaviour the live visuals depend on: no flicker, no
range collapse, no unbounded memory.
"""
from __future__ import annotations

import numpy as np

from sentrixviz import RollingNormalizer


def test_first_frame_seeds_range_exactly():
    rn = RollingNormalizer(alpha=0.2)
    lo, hi = rn.color_range("k", np.array([2.0, 5.0, 9.0]))
    assert (lo, hi) == (2.0, 9.0)


def test_expands_immediately_on_a_peak():
    """A sudden contact peak must NOT be clipped: the range jumps to cover it
    on the same frame it appears."""
    rn = RollingNormalizer(alpha=0.05)
    rn.color_range("k", np.array([0.0, 1.0]))
    lo, hi = rn.color_range("k", np.array([0.0, 50.0]))   # spike
    assert hi == 50.0           # expanded fully, not 1 + 0.05*(50-1)
    assert lo == 0.0


def test_relaxes_gradually_after_a_peak():
    """After the peak passes, the hi must drift DOWN gradually (no snap), so the
    colormap does not flicker - the gradual-adaptation requirement."""
    rn = RollingNormalizer(alpha=0.25)
    rn.color_range("k", np.array([0.0, 100.0]))           # establish high range
    his = []
    for _ in range(8):
        _, hi = rn.color_range("k", np.array([0.0, 10.0]))  # quiet frames
        his.append(hi)
    # strictly decreasing, monotone, and never below the live signal (10)
    assert all(b < a for a, b in zip(his, his[1:]))
    assert his[-1] > 10.0
    # each step is a bounded fraction of the gap (gradual, not a jump)
    assert his[0] < 100.0 and his[0] > 60.0


def test_converges_to_a_stationary_signal():
    rn = RollingNormalizer(alpha=0.5)
    last = None
    for _ in range(50):
        last = rn.color_range("k", np.array([3.0, 7.0]))
    lo, hi = last
    assert abs(lo - 3.0) < 1e-6 and abs(hi - 7.0) < 1e-6


def test_bounded_memory_over_a_long_stream():
    """State stays O(keys): rendering 5000 frames must not grow the policy."""
    rn = RollingNormalizer()
    for i in range(5000):
        rn.color_range("color", np.array([0.0, float(i % 13)]))
        rn.size_range("size", np.array([1.0, 2.0]))
        rn.vector_scale("vec", np.array([[1.0, 0.0], [0.0, 1.0]]))
    assert len(rn._color) == 1
    assert len(rn._size) == 1
    assert len(rn._vec_mag) == 1


def test_all_nan_frame_never_fabricates_a_range():
    rn = RollingNormalizer()
    assert rn.color_range("k", np.array([np.nan, np.nan])) == (None, None)
    # once a finite frame arrives, a later all-NaN frame keeps the last range
    rn.color_range("k", np.array([1.0, 4.0]))
    assert rn.color_range("k", np.array([np.nan])) == (1.0, 4.0)


def test_none_key_is_neutral():
    rn = RollingNormalizer()
    assert rn.color_range(None, np.array([1.0, 2.0])) == (None, None)
    assert rn.vector_scale(None, np.array([[1.0, 1.0]])) is None


def test_vector_scale_needs_a_primed_span():
    rn = RollingNormalizer()
    uv = np.array([[3.0, 4.0]])               # |uv| = 5
    assert rn.vector_scale("v", uv) is None    # no span yet -> autoscale fallback
    rn.prime_vector_span("v", span=10.0)
    scale = rn.vector_scale("v", uv)
    assert scale is not None and scale > 0


def test_categories_accumulate_and_stay_ordered():
    rn = RollingNormalizer()
    rn.observe_categories("c", ["a", "b"])
    rn.observe_categories("c", ["b", "c"])     # new label appended, no dupes
    assert rn.categories("c") == ["a", "b", "c"]
