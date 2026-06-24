"""Timeline semantics over a FrameSource: relative time, fps estimate, seek,
dropped-frame awareness, windowing. Runs across every descriptor case so no
assertion bakes in a sensor count or a topology.
"""
from __future__ import annotations

import math

from sentrixviz.sources import ParquetFrameSource
from sentrixviz.timeline import Timeline, TimelineFrame


def test_timeline_basic_metadata(raw_parquet):
    name, desc, path = raw_parquet
    tl = Timeline(ParquetFrameSource(path))
    assert len(tl) == 4
    # synth fixture: t = 0,1000,2000,3000 us -> 1000 fps, 3000 us span
    assert tl.t0_us == 0
    assert tl.duration_us == 3000
    assert math.isclose(tl.fps_estimate, 1000.0)
    assert tl.dropped() == []  # uniform cadence -> no gaps


def test_timeline_relative_time_and_order(raw_parquet):
    name, desc, path = raw_parquet
    tl = Timeline(ParquetFrameSource(path))
    frames = list(tl)
    assert [f.index for f in frames] == [0, 1, 2, 3]
    # timestamps strictly increasing; t_rel starts at 0
    ts = [f.t_capture_us for f in frames]
    assert ts == sorted(ts)
    assert frames[0].t_rel_us == 0
    assert all(f.t_rel_us == f.t_capture_us - tl.t0_us for f in frames)
    assert all(isinstance(f, TimelineFrame) for f in frames)


def test_timeline_seek_is_nearest(raw_parquet):
    name, desc, path = raw_parquet
    tl = Timeline(ParquetFrameSource(path))
    assert tl.seek(0) == 0
    assert tl.seek(2100) == 2          # nearest to t=2000
    assert tl.seek(10_000) == 3        # clamps to last


def test_timeline_window_stride(raw_parquet):
    name, desc, path = raw_parquet
    tl = Timeline(ParquetFrameSource(path))
    every_other = list(tl.window(step=2))
    assert [f.index for f in every_other] == [0, 2]
    sub = list(tl.window(start=1, stop=3))
    assert [f.index for f in sub] == [1, 2]


def test_dropped_frame_awareness(monkeypatch, raw_parquet):
    """A large timestamp gap is flagged as a likely drop; the median sets the
    baseline so the test is cadence-relative, not a hardcoded threshold."""
    name, desc, path = raw_parquet
    src = ParquetFrameSource(path)
    # inject a hole: shift the last frame far out (5x the 1000us cadence)
    real_ts = src.timestamps()
    holed = real_ts[:-1] + [real_ts[-1] + 5000]
    monkeypatch.setattr(src, "timestamps", lambda: holed)
    tl = Timeline(src)
    drops = tl.dropped()
    assert len(drops) == 1
    assert drops[0]["after_index"] == 2
    assert drops[0]["missing_estimate"] >= 1
