"""Phase 3 live path: descriptor / topology / count independence, replay parity,
and live-stream robustness - all on synthetic streams, no hardware.

The descriptor-parametrized fixtures (Mark2_v1 21+3, Mark2_tiny 7,
RobotFinger_v1 5) carry the independence guarantees by construction: every test
that runs across `descriptor_case` proves the live path is count- and
topology-blind, exactly like the playback suite.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from sentrixviz import RenderModel
from sentrixviz.layers import HeatmapLayer, ShearLayer
from sentrixviz.live import LiveSession
from sentrixviz.render import MatplotlibRenderer
from sentrixviz.sources import (
    LiveFrameSource,
    ParquetFrameSource,
    QueueFeed,
    RawFrame,
    ReplayFeed,
    SyntheticFeed,
)

from conftest import synth_raw_parquet


# ---- LiveFrameSource over a feed ---------------------------------------------

def test_live_source_exposes_descriptor_identity(descriptor_case):
    _, desc = descriptor_case
    live = LiveFrameSource(SyntheticFeed(desc, n_frames=3))
    assert live.descriptor_version == desc.descriptor_version
    assert live.descriptor_hash == (desc.descriptor_hash or None)
    assert live.ts_column == "t_capture_us"


def test_synthetic_feed_yields_fieldsets_with_increasing_time(descriptor_case):
    _, desc = descriptor_case
    live = LiveFrameSource(SyntheticFeed(desc, n_frames=5))
    frames = list(live)
    assert len(frames) == 5
    ts = [f.t_capture_us for f in frames]
    assert ts == sorted(ts) and ts[0] == 0
    # channels were discovered from the stream (count-blind)
    assert live.channels()                       # non-empty after iteration


# ---- replay parity: the live path matches playback exactly -------------------

def test_replay_feed_roundtrips_to_playback_fieldsets(descriptor_case, tmp_path):
    """ReplayFeed -> LiveFrameSource must yield FieldSets byte-equal to
    ParquetFrameSource.frame(i). Proves the live code path produces the same
    FieldSet as playback (no second schema, no drift)."""
    name, desc = descriptor_case
    path = synth_raw_parquet(desc, tmp_path / f"{name}.parquet", n_frames=6)
    src = ParquetFrameSource(path)
    live = LiveFrameSource(ReplayFeed(src))

    for i, fs_live in enumerate(live):
        fs_play = src.frame(i)
        assert fs_live.t_capture_us == fs_play.t_capture_us
        assert fs_live.frame_index == fs_play.frame_index
        assert fs_live.valid == fs_play.valid
        assert fs_live.values.keys() == fs_play.values.keys()
        for sid, chans in fs_play.values.items():
            for ch, v in chans.items():
                a, b = fs_live.values[sid][ch], v
                assert (a == b) or (math.isnan(a) and math.isnan(b))


# ---- LiveSession drives the existing stack -----------------------------------

@pytest.mark.parametrize("view,layer_factory", [
    ("heatmap", lambda: [HeatmapLayer()]),
    ("shear", lambda: [ShearLayer()]),
])
def test_live_session_renders_every_descriptor(descriptor_case, tmp_path, view, layer_factory):
    """One live code path renders any descriptor / topology / count to frames."""
    name, desc = descriptor_case
    model = RenderModel.from_descriptor(desc)
    live = LiveFrameSource(SyntheticFeed(desc, n_frames=8))
    out_dir = tmp_path / f"{name}_{view}"
    session = LiveSession(model, layer_factory(), MatplotlibRenderer(), view=view)
    manifest = session.run(live, out_dir=out_dir, max_frames=8)

    assert manifest["n_frames"] == 8
    pngs = sorted(out_dir.glob("*.png"))
    assert len(pngs) == 8
    assert all(p.stat().st_size > 0 for p in pngs)
    saved = json.loads((out_dir / "manifest.json").read_text())
    assert saved["mode"] == "live"
    assert saved["descriptor_version"] == desc.descriptor_version
    # the manifest carries the sync key per frame (video-overlay forward compat)
    assert all("t_capture_us" in f for f in saved["frames"])


def test_live_session_latest_wins_monitor(descriptor_case, tmp_path):
    """`out` mode overwrites a single file each frame -> a live monitor."""
    name, desc = descriptor_case
    model = RenderModel.from_descriptor(desc)
    live = LiveFrameSource(SyntheticFeed(desc, n_frames=5))
    out = tmp_path / "latest.png"
    LiveSession(model, [HeatmapLayer()], MatplotlibRenderer()).run(
        live, out=out, max_frames=5)
    assert out.exists() and out.stat().st_size > 0


def test_max_frames_stops_an_unbounded_stream(descriptor_case, tmp_path):
    _, desc = descriptor_case
    model = RenderModel.from_descriptor(desc)
    live = LiveFrameSource(SyntheticFeed(desc, n_frames=None))   # infinite
    out_dir = tmp_path / "bounded"
    manifest = LiveSession(model, [HeatmapLayer()], MatplotlibRenderer()).run(
        live, out_dir=out_dir, max_frames=4)
    assert manifest["n_frames"] == 4


# ---- robustness ---------------------------------------------------------------

def test_dropout_frame_renders_without_crashing(descriptor_case, tmp_path):
    """A fully-invalid (NaN) frame must render (hollow markers) not crash -
    hard rule #5, never fabricate a gap."""
    name, desc = descriptor_case
    model = RenderModel.from_descriptor(desc)

    class _DropoutFeed:
        descriptor_version = desc.descriptor_version
        descriptor_hash = desc.descriptor_hash or None
        ts_column = "t_capture_us"

        def __iter__(self):
            from sentrix_contracts import column_for
            for i in range(3):
                row = {}
                for s in desc.sensors.values():
                    for ch in s.channels:
                        row[column_for(s.sensor_id, ch)] = float("nan")
                    row[f"{s.sensor_id}.valid"] = False
                yield RawFrame(row, t_capture_us=i, frame_index=i)

    live = LiveFrameSource(_DropoutFeed())
    out_dir = tmp_path / "drop"
    manifest = LiveSession(model, [HeatmapLayer()], MatplotlibRenderer()).run(
        live, out_dir=out_dir, max_frames=3)
    assert manifest["n_frames"] == 3
    assert len(list(out_dir.glob("*.png"))) == 3


def test_queue_feed_is_latest_wins_and_never_blocks():
    """A bounded QueueFeed drops the OLDEST frame when full (never backpressures
    the producer) and ends cleanly on close()."""
    feed = QueueFeed("Synthetic_v1", capacity=2)
    for i in range(5):
        feed.push(RawFrame({}, t_capture_us=i, frame_index=i))   # never blocks
    feed.close()
    drained = list(feed)
    assert feed.dropped == 3                     # 5 pushed, capacity 2
    assert [f.frame_index for f in drained] == [3, 4]   # freshest kept


def test_queue_feed_threaded_producer_consumer():
    """Producer thread pushes faster than the consumer renders; consumer still
    drains the freshest frames and the producer never blocks."""
    import threading

    feed = QueueFeed("Synthetic_v1", capacity=1)
    received: list[int] = []

    def produce():
        for i in range(20):
            feed.push(RawFrame({}, t_capture_us=i, frame_index=i))
        feed.close()

    t = threading.Thread(target=produce)
    t.start()
    for raw in feed:
        received.append(raw.frame_index)
    t.join()
    # we cannot predict exactly which survived the latest-wins ring, but every
    # delivered frame is a real one and the stream terminated.
    assert all(0 <= r < 20 for r in received)
    assert feed.dropped + len(received) == 20
