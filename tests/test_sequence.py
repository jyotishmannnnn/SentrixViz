"""render_sequence: ordered PNGs + manifest, across every descriptor and every
view. Asserts ordering, timestamp ordering, descriptor independence, and count
independence - nothing here knows how many sensors or what topology a case has.
"""
from __future__ import annotations

import json

import pytest

from sentrixviz.core.model import RenderModel
from sentrixviz.layers import (
    CentroidLayer,
    ClusterLayer,
    FieldLayer,
    HeatmapLayer,
    ShearLayer,
)
from sentrixviz.render import MatplotlibRenderer
from sentrixviz.sources import ParquetFrameSource
from sentrixviz.timeline import Timeline, render_sequence

VIEWS = {
    "raw": lambda: [FieldLayer(reduction="mag")],
    "heatmap": lambda: [HeatmapLayer(reduction="mag")],
    "clusters": lambda: [ClusterLayer(reduction="mag")],
    "centroid": lambda: [HeatmapLayer(reduction="mag"), CentroidLayer(reduction="mag")],
    "shear": lambda: [ShearLayer(reduction="mag")],
}


@pytest.mark.parametrize("view", list(VIEWS))
def test_sequence_generates_ordered_pngs(raw_parquet, tmp_path, view):
    name, desc, path = raw_parquet
    model = RenderModel.from_descriptor(desc)
    tl = Timeline(ParquetFrameSource(path))
    out_dir = tmp_path / f"{name}_{view}"
    manifest = render_sequence(
        model, VIEWS[view](), tl, out_dir, MatplotlibRenderer(),
        view=view, channel="mag",
    )
    # one PNG per frame, zero-padded, contiguous from 000000
    pngs = sorted(p.name for p in out_dir.glob("[0-9]*.png"))
    assert pngs == [f"{i:06d}.png" for i in range(len(tl))]
    assert manifest["n_frames"] == len(tl)
    assert all((out_dir / f["file"]).stat().st_size > 0 for f in manifest["frames"])


def test_sequence_frame_and_timestamp_ordering(raw_parquet, tmp_path):
    name, desc, path = raw_parquet
    model = RenderModel.from_descriptor(desc)
    tl = Timeline(ParquetFrameSource(path))
    manifest = render_sequence(model, [FieldLayer(reduction="mag")], tl,
                               tmp_path / name, MatplotlibRenderer())
    frames = manifest["frames"]
    # emission order matches file order matches increasing index & timestamp
    assert [f["index"] for f in frames] == sorted(f["index"] for f in frames)
    ts = [f["t_capture_us"] for f in frames]
    assert ts == sorted(ts)
    assert frames[0]["t_rel_us"] == 0


def test_manifest_written_and_self_describing(raw_parquet, tmp_path):
    name, desc, path = raw_parquet
    model = RenderModel.from_descriptor(desc)
    tl = Timeline(ParquetFrameSource(path))
    out_dir = tmp_path / name
    render_sequence(model, [HeatmapLayer(reduction="mag")], tl, out_dir,
                    MatplotlibRenderer(), view="heatmap", channel="mag")
    loaded = json.loads((out_dir / "manifest.json").read_text())
    assert loaded["descriptor_version"] == desc.descriptor_version
    assert loaded["view"] == "heatmap"
    assert loaded["n_frames"] == len(tl)
    assert len(loaded["frames"]) == len(tl)


def test_step_decimates(raw_parquet, tmp_path):
    name, desc, path = raw_parquet
    model = RenderModel.from_descriptor(desc)
    tl = Timeline(ParquetFrameSource(path))
    manifest = render_sequence(model, [FieldLayer(reduction="mag")], tl,
                               tmp_path / name, MatplotlibRenderer(), step=2)
    # 4 frames, stride 2 -> emit indices 0 and 2, renamed densely 0,1
    assert [f["file"] for f in manifest["frames"]] == ["000000.png", "000001.png"]
    assert [f["index"] for f in manifest["frames"]] == [0, 2]


def test_count_independence_across_descriptors(raw_parquet, tmp_path):
    """Same code path, different sensor counts/topologies -> same contract:
    one ordered PNG per frame. No assertion references a count."""
    name, desc, path = raw_parquet
    model = RenderModel.from_descriptor(desc)
    tl = Timeline(ParquetFrameSource(path))
    manifest = render_sequence(model, [HeatmapLayer(reduction="mag")], tl,
                               tmp_path / name, MatplotlibRenderer())
    assert manifest["n_frames"] == len(tl)
    assert manifest["device_class"] == model.device_class
