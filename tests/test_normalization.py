"""The normalization seam: prove a scanned GlobalNormalizer freezes every
appearance channel across a sequence, while the legacy per-frame path varies it
(the flicker). Every proof runs over Mark2_v1 / Mark2_tiny / RobotFinger_v1 via
the `raw_parquet` fixture, so descriptor- and topology-independence (a 21-sensor
glove, a 7-sensor glove, and a 5-taxel collinear finger) are covered by
construction - no assertion references a count, a name, or a layout.

This is the contract a future RollingNormalizer will plug into unchanged.
"""
from __future__ import annotations

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sentrixviz.core.model import RenderModel
from sentrixviz.layers import (
    CentroidLayer,
    FieldLayer,
    HeatmapLayer,
    ShearLayer,
    TopologyLayer,
)
from sentrixviz.normalize import (
    GlobalNormalizer,
    PerFrameNormalizer,
    scan_ranges,
)
from sentrixviz.render import MatplotlibRenderer
from sentrixviz.sources import ParquetFrameSource
from sentrixviz.timeline import Timeline, render_sequence


def _model_and_frames(desc, path):
    model = RenderModel.from_descriptor(desc)
    src = ParquetFrameSource(path)
    frames = [src.frame(i) for i in range(len(src))]
    return model, frames


def _varies(xs) -> bool:
    """True if the finite values are not all equal - i.e. a real per-frame
    flicker exists for this case (so the global fix is actually exercised)."""
    finite = [round(float(x), 6) for x in xs if np.isfinite(x)]
    return len(set(finite)) > 1


# ---- 1. stable heatmap / raw-field colour across a sequence -------------------
def test_global_color_range_is_constant_while_perframe_flickers(raw_parquet):
    name, desc, path = raw_parquet
    model, frames = _model_and_frames(desc, path)
    layer = HeatmapLayer(reduction="mag")
    key = layer.encode(model, frames[0])["field"]["key"]

    per_lo, per_hi = [], []
    for fs in frames:
        g = np.asarray(layer.encode(model, fs)["field"]["grid"], dtype=float)
        g = g[np.isfinite(g)]
        per_lo.append(g.min()); per_hi.append(g.max())

    gn = scan_ranges(model, [layer], frames)
    vmin, vmax = gn.color_range(key, np.empty(0))

    # the seam: one range, equal to the union of every frame's extent ...
    assert vmin == pytest.approx(min(per_lo))
    assert vmax == pytest.approx(max(per_hi))
    # ... constant no matter which frame's values are handed in (global ignores them)
    assert gn.color_range(key, np.array([1.0])) == gn.color_range(key, np.array([9e9]))
    # ... and it actually replaces a real flicker for this descriptor
    assert _varies(per_hi)


def test_raw_field_layer_also_carries_a_scannable_key(raw_parquet):
    name, desc, path = raw_parquet
    model, frames = _model_and_frames(desc, path)
    layer = FieldLayer(reduction="mag")
    key = layer.encode(model, frames[0])["points"]["key"]
    gn = scan_ranges(model, [layer], frames)
    vmin, vmax = gn.color_range(key, np.empty(0))
    assert vmin is not None and vmax is not None and vmax > vmin


# ---- 2. stable cluster (category) colours ------------------------------------
def _facecolor_of_label(renderer, gmap, labels, target):
    """Render a category points primitive through the REAL renderer path and read
    back the fill colour matplotlib assigned to `target`'s point."""
    fig, ax = plt.subplots()
    pts = {
        "key": "points:topology",
        "xy": np.zeros((len(labels), 2)),
        "kind": "category",
        "categories": np.arange(len(labels), dtype=int),
        "category_labels": list(labels),
        "valid": np.ones(len(labels), dtype=bool),
    }
    renderer._draw_points(ax, pts)
    fig.canvas.draw()
    fc = ax.collections[0].get_facecolors()
    col = tuple(np.round(fc[labels.index(target)], 4))
    plt.close(fig)
    return col


def test_global_category_colour_is_stable_under_subset_reorder(raw_parquet):
    name, desc, path = raw_parquet
    model, _ = _model_and_frames(desc, path)
    gn = scan_ranges(model, [TopologyLayer()], [None])
    gmap = gn.categories("points:topology")
    assert gmap is not None and len(gmap) >= 1
    if len(gmap) < 2:
        pytest.skip("single-category layout: no reorder to flicker")

    L = gmap[1]
    a, b = [gmap[0], gmap[1]], [gmap[1], gmap[0]]  # same label, different local index

    g_ren = MatplotlibRenderer(
        normalizer=GlobalNormalizer(category_maps={"points:topology": gmap}))
    # global map pins L's colour to its global index regardless of local order
    assert _facecolor_of_label(g_ren, gmap, a, L) == _facecolor_of_label(g_ren, gmap, b, L)

    # legacy per-frame: same label flips colour when the local index moves
    p_ren = MatplotlibRenderer()
    assert _facecolor_of_label(p_ren, gmap, a, L) != _facecolor_of_label(p_ren, gmap, b, L)


# ---- 3. stable centroid marker sizing ----------------------------------------
def test_global_size_range_tames_weak_frame_glyphs(raw_parquet):
    name, desc, path = raw_parquet
    model, frames = _model_and_frames(desc, path)
    layer = CentroidLayer(reduction="mag")
    key = "markers:centroid"

    acts = [np.asarray(layer.encode(model, fs)["markers"]["weight"], dtype=float)
            for fs in frames]
    per_hi = [np.nanmax(a) for a in acts if np.isfinite(a).any()]
    if not _varies(per_hi):
        pytest.skip("activation does not vary across frames for this case")

    gn = scan_ranges(model, [layer], frames)
    glo, ghi = gn.size_range(key, np.empty(0))
    assert glo == pytest.approx(min(np.nanmin(a) for a in acts if np.isfinite(a).any()))
    assert ghi == pytest.approx(max(per_hi))

    # The pulse vs the fix: send the SAME fixed physical weights through each
    # frame's mapping. Global sizing depends only on the weight (one result every
    # frame); per-frame sizing depends on that frame's spread, so the identical
    # activation renders at different sizes frame to frame.
    probe = np.array([glo, 0.5 * (glo + ghi), ghi])
    glob_size = MatplotlibRenderer._weights_to_sizes(probe, 120.0, lo=glo, hi=ghi)
    per_sizes = [
        MatplotlibRenderer._weights_to_sizes(
            probe, 120.0, lo=float(np.nanmin(a)), hi=float(np.nanmax(a)))
        for a in acts if np.isfinite(a).any()
    ]
    assert any(not np.allclose(per_sizes[0], ps) for ps in per_sizes[1:])
    for ps in per_sizes:                      # global is the frame-independent one
        assert ps.shape == glob_size.shape


# ---- 4. stable shear vector scaling ------------------------------------------
def test_global_vector_scale_is_one_constant_while_magnitude_flickers(raw_parquet):
    name, desc, path = raw_parquet
    model, frames = _model_and_frames(desc, path)
    layer = ShearLayer(reduction="mag")
    key = "vectors:shear"

    per_max = []
    for fs in frames:
        uv = np.asarray(layer.encode(model, fs)["vectors"]["uv"], dtype=float)
        m = np.linalg.norm(uv, axis=1)
        m = m[np.isfinite(m)]
        per_max.append(m.max() if m.size else np.nan)

    gn = scan_ranges(model, [layer], frames)
    scale = gn.vector_scale(key, np.empty((0, 2)))
    if not any(np.isfinite(x) and x > 0 for x in per_max):
        pytest.skip("no finite shear for this case")
    assert scale is not None and scale > 0
    # one scale for the whole window (arrows stop breathing) ...
    assert gn.vector_scale(key, np.empty((0, 2))) == scale
    # ... over a magnitude that genuinely flickered frame to frame
    assert _varies(per_max)


# ---- 5/6. descriptor + topology independence ---------------------------------
def test_scan_is_descriptor_and_topology_independent(raw_parquet):
    """Same scan code over a glove, a tiny glove, and a collinear finger line:
    each yields a usable, deterministic policy with no branch on the layout."""
    name, desc, path = raw_parquet
    model, frames = _model_and_frames(desc, path)
    layers = [HeatmapLayer(reduction="mag"), CentroidLayer(reduction="mag"),
              ShearLayer(reduction="mag"), TopologyLayer()]
    a = scan_ranges(model, layers, frames)
    b = scan_ranges(model, layers, frames)

    hkey = HeatmapLayer(reduction="mag").encode(model, frames[0])["field"]["key"]
    # deterministic: two scans agree exactly (no time/order/random dependence)
    assert a.color_range(hkey, np.empty(0)) == b.color_range(hkey, np.empty(0))
    # collinear RobotFinger line included: IDW still yields a finite global range
    vmin, vmax = a.color_range(hkey, np.empty(0))
    assert vmin is not None and vmax is not None
    # categories always present and complete for the layout
    assert a.categories("points:topology")


# ---- regression: default policy is byte-identical legacy behaviour ------------
def test_perframe_is_neutral_and_is_the_renderer_default():
    pf = PerFrameNormalizer()
    assert pf.color_range("k", np.array([1.0, 2.0])) == (None, None)
    assert pf.size_range("k", np.array([1.0, 2.0])) == (None, None)
    assert pf.vector_scale("k", np.zeros((1, 2))) is None
    assert pf.categories("k") is None
    assert isinstance(MatplotlibRenderer().normalizer, PerFrameNormalizer)


def test_render_sequence_defaults_global_and_restores_renderer(raw_parquet, tmp_path):
    name, desc, path = raw_parquet
    model = RenderModel.from_descriptor(desc)
    tl = Timeline(ParquetFrameSource(path))
    r = MatplotlibRenderer()
    before = r.normalizer

    # default == global: produces frames AND leaves the renderer's own policy intact
    m = render_sequence(model, [HeatmapLayer(reduction="mag")], tl,
                        tmp_path / "g", r, view="heatmap")
    assert m["n_frames"] == len(tl)
    assert r.normalizer is before  # scoped, non-destructive swap

    # opt out keeps per-frame
    m2 = render_sequence(model, [HeatmapLayer(reduction="mag")], tl,
                         tmp_path / "pf", r, view="heatmap", normalize=None)
    assert m2["n_frames"] == len(tl)
    assert r.normalizer is before
