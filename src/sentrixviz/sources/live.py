"""Live (push-stream) frame production - the Phase 3 sibling of ParquetFrameSource.

The frozen `FrameSource` protocol is finite and random-access (`__len__`,
`frame(i)`, `timestamps()`). A live stream is unbounded and push-based: no
length, no past. So Phase 3 does NOT pretend a live stream is a FrameSource;
it adds a push-mode producer with the SAME philosophy (descriptor-paired, emits
`(timestamp, FieldSet)`, backend-blind) and reuses everything downstream
(Layer/Renderer/NormalizationPolicy only ever need a FieldSet).

The single new abstraction is `FrameFeed`: a transport seam yielding `RawFrame`
records (the shared at-rest column convention + timestamp + frame_index). The
transport - USB, socket, an in-process Capture bus, a Sim stream - lives BEHIND
the feed and is never imported here, exactly as pyarrow lives behind the parquet
source. This is what keeps Viz hardware-independent (hard rule #3).

What this module must NEVER do:
  - import SentrixCapture / Sim / Sync / a USB driver,
  - assume a sensor count, a channel set, or a fixed ts column,
  - block (backpressure) a producer: the live tap is latest-wins.
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Iterator, Protocol, runtime_checkable

from sentrix_contracts import column_for, parse_column

from ..core.fields import FieldSet, bind_row
from .parquet import ParquetFrameSource


@dataclass(frozen=True)
class RawFrame:
    """One frame as it arrives off a transport: the at-rest column->value row
    (recognized by parse_column, identical to a parquet row), plus its capture
    timestamp and the producer's frame counter. This is what every feed emits;
    LiveFrameSource turns it into a FieldSet with the existing bind_row."""
    row: dict[str, object]
    t_capture_us: int
    frame_index: int


@runtime_checkable
class FrameFeed(Protocol):
    """A descriptor-paired, push-mode stream of RawFrames. The transport seam.

    A feed knows its descriptor identity + ts column up front (a stream header /
    handshake) and yields frames until the stream ends. It NEVER knows positions,
    topology, or how anything is drawn - same boundary as FrameSource."""
    descriptor_version: str
    descriptor_hash: str | None
    ts_column: str

    def __iter__(self) -> Iterator[RawFrame]: ...


class LiveFrameSource:
    """Iterable, descriptor-paired producer of FieldSets over a FrameFeed.

    Same interface philosophy as ParquetFrameSource (exposes descriptor identity
    + ts_column + channels, binds rows with the same bind_row) but UNBOUNDED:
    iterate it, do not index it. `resolve_descriptor` works on it unchanged
    because it only reads descriptor_version / descriptor_hash.
    """

    def __init__(self, feed: FrameFeed):
        self._feed = feed
        self._channels: set[str] = set()

    # ---- identity (mirrors ParquetFrameSource) --------------------------
    @property
    def descriptor_version(self) -> str:
        return self._feed.descriptor_version

    @property
    def descriptor_hash(self) -> str | None:
        return self._feed.descriptor_hash or None

    @property
    def ts_column(self) -> str:
        return self._feed.ts_column

    def channels(self) -> list[str]:
        """Channels seen so far (grows as frames arrive). Empty before the first
        frame - a live stream cannot know its channels until one is observed."""
        return sorted(self._channels)

    # ---- push consumption ----------------------------------------------
    def __iter__(self) -> Iterator[FieldSet]:
        for raw in self._feed:
            for col in raw.row:
                parsed = parse_column(col)
                if parsed is not None:
                    self._channels.add(parsed[1])
            yield bind_row(raw.row, t_capture_us=raw.t_capture_us,
                           frame_index=raw.frame_index)


# ---- built-in feeds (no hardware) ---------------------------------------------

def _row_from_fieldset(fs: FieldSet) -> dict[str, object]:
    """Reverse of bind_row: a FieldSet back to an at-rest row. Lets a recorded
    artifact be replayed AS a live stream through the exact live code path
    (so the live path is testable without hardware, and provably matches
    playback). Uses only column_for - no second schema."""
    row: dict[str, object] = {}
    for sid, chans in fs.values.items():
        for ch, val in chans.items():
            row[column_for(sid, ch)] = val
    for sid, ok in fs.valid.items():
        row[f"{sid}.valid"] = ok
    return row


class ReplayFeed:
    """Replays a finite FrameSource (e.g. a recorded raw.parquet) as a live
    push stream. The bridge that lets the whole live stack be exercised with no
    hardware, and the parity oracle for tests (live FieldSet == playback frame)."""

    def __init__(self, source: ParquetFrameSource, *, loop: bool = False):
        self._src = source
        self._loop = loop

    @property
    def descriptor_version(self) -> str:
        return self._src.descriptor_version

    @property
    def descriptor_hash(self) -> str | None:
        return self._src.descriptor_hash

    @property
    def ts_column(self) -> str:
        return self._src.ts_column

    def __iter__(self) -> Iterator[RawFrame]:
        while True:
            for i in range(len(self._src)):
                fs = self._src.frame(i)
                yield RawFrame(_row_from_fieldset(fs),
                               t_capture_us=fs.t_capture_us,
                               frame_index=fs.frame_index)
            if not self._loop:
                return


class SyntheticFeed:
    """Descriptor-driven synthetic live stream - a moving contact bump - for
    demos and tests with no producer and no hardware. Count- and topology-blind:
    it emits exactly the channels each descriptor sensor declares, for any
    sensor count and any layout. Deterministic (no RNG) so tests are stable."""

    def __init__(self, desc, *, n_frames: int | None = 16, ts_step_us: int = 1000):
        self._desc = desc
        self._n = n_frames
        self._dt = ts_step_us
        self.descriptor_version = desc.descriptor_version
        self.descriptor_hash = desc.descriptor_hash or None
        self.ts_column = "t_capture_us"
        # stable sensor order; index drives the moving bump (no name parsing)
        self._sensors = list(desc.sensors.values())

    def _value(self, sensor_index: int, n_sensors: int, ch: str, frame: int) -> float:
        import math
        # a Gaussian bump whose centre sweeps across the sensor index each frame
        centre = (frame % max(n_sensors, 1)) if n_sensors else 0
        d = sensor_index - centre
        bump = math.exp(-(d * d) / 2.0)            # 0..1 contact strength
        phase = 0.3 * frame + sensor_index
        table = {
            "bx": 5.0 * bump * math.cos(phase),
            "by": 5.0 * bump * math.sin(phase),
            "bz": 40.0 * bump,
            "ax": 2.0 * bump * math.cos(phase),
            "ay": 2.0 * bump * math.sin(phase),
            "az": 9.8,
            "temp": 25.0,
        }
        return float(table.get(ch, bump))

    def __iter__(self) -> Iterator[RawFrame]:
        n = len(self._sensors)
        i = 0
        while self._n is None or i < self._n:
            row: dict[str, object] = {}
            for si, s in enumerate(self._sensors):
                for ch in s.channels:
                    row[column_for(s.sensor_id, ch)] = self._value(si, n, ch, i)
                row[f"{s.sensor_id}.valid"] = True
            yield RawFrame(row, t_capture_us=i * self._dt, frame_index=i)
            i += 1


@dataclass
class QueueFeed:
    """The canonical live tap: a thread-safe, bounded, latest-wins ring a real
    transport (socket / USB reader / in-process Capture bus) pushes into.

    `push` NEVER blocks - when the ring is full it drops the OLDEST frame
    (counted in `dropped`), so the feed can never backpressure the recorder
    (CLAUDE.md P3 note). Iterating consumes the freshest available frames and
    ends cleanly after `close()`.
    """
    descriptor_version: str
    descriptor_hash: str | None = None
    ts_column: str = "t_capture_us"
    capacity: int = 1
    dropped: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._buf: deque[RawFrame] = deque(maxlen=max(1, self.capacity))
        self._cond = threading.Condition()
        self._closed = False

    def push(self, frame: RawFrame) -> None:
        with self._cond:
            if len(self._buf) == self._buf.maxlen:
                self.dropped += 1          # deque(maxlen) evicts the oldest
            self._buf.append(frame)
            self._cond.notify()

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def __iter__(self) -> Iterator[RawFrame]:
        while True:
            with self._cond:
                while not self._buf and not self._closed:
                    self._cond.wait()
                if not self._buf and self._closed:
                    return
                frame = self._buf.popleft()
            yield frame


__all__ = [
    "RawFrame", "FrameFeed", "LiveFrameSource",
    "ReplayFeed", "SyntheticFeed", "QueueFeed",
]
