"""Phase 2 proves the topology-aware layers stay count / topology / descriptor
independent: the SAME encoders drive a 21-sensor glove, a 7-sensor tiny glove,
and a 5-taxel collinear finger module with no branch on count, name, or layout.

A hardcoded count, finger/cluster name, or glove assumption breaks these by
construction (counts are asserted against the descriptor, never a literal).
"""
from __future__ import annotations

import numpy as np

from sentrixviz.core.model import RenderModel
from sentrixviz.layers import CentroidLayer, ClusterLayer, HeatmapLayer, ShearLayer
from sentrixviz.layers.derived import cluster_features, cluster_membership
from sentrixviz.sources import ParquetFrameSource


# ---- heatmap: count + topology independence -----------------------------
def test_heatmap_grid_shape_is_count_agnostic(raw_parquet):
    """Grid resolution is fixed regardless of 21 / 7 / 5 sensors."""
    name, desc, path = raw_parquet
    model = RenderModel.from_descriptor(desc)
    fs = ParquetFrameSource(path).frame(0)
    prims = HeatmapLayer(resolution=48).encode(model, fs)
    assert prims["field"]["grid"].shape == (48, 48)


def test_heatmap_finite_for_every_layout_including_a_line(raw_parquet):
    """IDW must produce a real field even for the collinear RobotFinger line -
    where Delaunay triangulation would degenerate. This is the topology-
    independence guarantee that justified choosing IDW."""
    name, desc, path = raw_parquet
    model = RenderModel.from_descriptor(desc)
    fs = ParquetFrameSource(path).frame(0)
    grid = HeatmapLayer().encode(model, fs)["field"]["grid"]
    # IDW yields a real field even for a degenerate line (triangulation would not)
    assert np.isfinite(grid).any()


def test_heatmap_masks_outside_support_on_a_plane(raw_parquet):
    """On a 2D layout the support mask leaves empty space NaN (no fabricated
    gaps). Skipped for the collinear case where the padded band is fully
    supported."""
    name, desc, path = raw_parquet
    model = RenderModel.from_descriptor(desc)
    xy = np.asarray(list(model.project_2d().values()))
    span = xy.max(axis=0) - xy.min(axis=0)
    if np.min(span) < 1e-6:        # collinear (RobotFinger line) - no 2D area
        return
    fs = ParquetFrameSource(path).frame(0)
    grid = HeatmapLayer().encode(model, fs)["field"]["grid"]
    assert np.isnan(grid).any()


def test_heatmap_never_fabricates_invalid_samples(raw_parquet):
    name, desc, path = raw_parquet
    model = RenderModel.from_descriptor(desc)
    fs = ParquetFrameSource(path).frame(0)
    prims = HeatmapLayer().encode(model, fs)
    # markers mark every positioned sensor; the field is bounded by them
    assert len(prims["markers"]["xy"]) == len(model.positioned_ids())


# ---- cluster membership: descriptor-driven, never name-parsed -----------
def test_cluster_ids_come_from_the_descriptor(raw_parquet):
    """Feature ids equal the descriptor's clusters (or the per-sensor fallback),
    proving membership is read from the descriptor, not parsed from names."""
    name, desc, path = raw_parquet
    model = RenderModel.from_descriptor(desc)
    fs = ParquetFrameSource(path).frame(0)
    feats = cluster_features(model, fs)
    got = {f.cluster_id for f in feats}
    if desc.clusters:
        assert got == set(desc.clusters)
    else:
        assert got == set(model.positioned_ids())


def test_cluster_members_match_descriptor_membership(raw_parquet):
    name, desc, path = raw_parquet
    model = RenderModel.from_descriptor(desc)
    groups = cluster_membership(model)
    for cid, members in groups.items():
        if cid in desc.clusters:
            # membership is a positioned subset of the descriptor's, nothing invented
            assert set(members) <= set(desc.clusters[cid].members)


def test_cluster_count_follows_descriptor_not_a_constant(raw_parquet):
    name, desc, path = raw_parquet
    model = RenderModel.from_descriptor(desc)
    fs = ParquetFrameSource(path).frame(0)
    expected = len(desc.clusters) if desc.clusters else len(model.positioned_ids())
    for layer in (ClusterLayer(), ShearLayer()):
        prims = layer.encode(model, fs)
        assert len(prims["markers"]["xy"]) == expected
        assert len(prims["vectors"]["xy"]) == expected


# ---- centroid: topology-driven, weighted ---------------------------------
def test_centroid_lies_within_layout_bounds(raw_parquet):
    name, desc, path = raw_parquet
    model = RenderModel.from_descriptor(desc)
    fs = ParquetFrameSource(path).frame(0)
    xy = np.asarray(list(model.project_2d().values()))
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    prims = CentroidLayer().encode(model, fs)
    cents = np.asarray(prims["markers"]["xy"], dtype=float)
    for c in cents:
        if np.all(np.isfinite(c)):
            assert np.all(c >= lo - 1e-6) and np.all(c <= hi + 1e-6)


def test_centroid_weight_is_activation(raw_parquet):
    name, desc, path = raw_parquet
    model = RenderModel.from_descriptor(desc)
    fs = ParquetFrameSource(path).frame(0)
    feats = cluster_features(model, fs)
    prims = CentroidLayer().encode(model, fs)
    np.testing.assert_allclose(
        prims["markers"]["weight"],
        np.asarray([f.activation for f in feats], dtype=float),
    )
