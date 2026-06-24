"""Timeline - a read-only, metadata-bearing view over any FrameSource.

L2.5 of the SDK. It introduces NO new contract: a Timeline is built from a
FrameSource (the frozen Phase-1 stream contract) and adds only the timeline
semantics that do not belong in a data source - relative time, an FPS estimate,
dropped-frame awareness, windowing, and a uniform per-frame record. Because it
depends only on the FrameSource protocol, a future SyncFrameSource (as-of join
of many sources) drives it with no change here.

What this module must NEVER do:
  - hold positions / topology (that is RenderModel),
  - import a renderer or a layer (orchestration composes those, not Timeline),
  - assume a sensor count, a channel set, or a fixed cadence.

Seeking is the source's `index_at_time`; frame indexing is an integer index.
There is deliberately no play/pause/speed controller - that is a live-frontend
concern and is out of Phase 2.5 scope.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from ..core.fields import FieldSet
from ..core.protocols import FrameSource


@dataclass(frozen=True)
class TimelineFrame:
    """One materialized frame plus its timeline coordinates.

    `index`        - position in the source (0-based, contiguous).
    `frame_index`  - the producer's own frame counter (may have gaps == drops).
    `t_capture_us` - absolute capture timestamp (the camera/video sync key).
    `t_rel_us`     - time since the first frame (playback clock).
    """
    index: int
    frame_index: int
    t_capture_us: int
    t_rel_us: int
    fieldset: FieldSet


class Timeline:
    """Read-only timeline over a FrameSource. Cheap to build (uses only the
    source's cached timestamps); frames are materialized lazily on access."""

    def __init__(self, source: FrameSource):
        self._src = source
        self._ts = list(source.timestamps())
        if len(self._ts) != len(source):
            # A well-formed source reports one timestamp per frame; refuse to
            # guess if it does not (no fabricated cadence).
            raise ValueError(
                f"timestamp/frame count mismatch: {len(self._ts)} ts vs "
                f"{len(source)} frames"
            )

    # ---- size / time ----------------------------------------------------
    def __len__(self) -> int:
        return len(self._src)

    @property
    def t0_us(self) -> int:
        return self._ts[0] if self._ts else 0

    @property
    def duration_us(self) -> int:
        return (self._ts[-1] - self._ts[0]) if len(self._ts) > 1 else 0

    def timestamps(self) -> list[int]:
        return list(self._ts)

    @property
    def fps_estimate(self) -> float:
        """Frames per second from the median inter-frame interval (robust to a
        few large gaps). 0.0 when undefined (< 2 frames or zero spread)."""
        dt = self._intervals()
        if not dt:
            return 0.0
        med = sorted(dt)[len(dt) // 2]
        return (1_000_000.0 / med) if med > 0 else 0.0

    # ---- dropped-frame awareness ----------------------------------------
    def dropped(self, factor: float = 1.5) -> list[dict]:
        """Intervals whose gap exceeds `factor` x the median interval, i.e. a
        likely dropped frame. Returns the boundary indices and the gap so a
        consumer (or a future sync tool) can decide how to treat the hole. We
        flag; we never fabricate the missing frame."""
        dt = self._intervals()
        if len(dt) < 2:
            return []
        med = sorted(dt)[len(dt) // 2]
        if med <= 0:
            return []
        out: list[dict] = []
        for i, gap in enumerate(dt):
            if gap > factor * med:
                out.append({
                    "after_index": i,
                    "gap_us": int(gap),
                    "missing_estimate": max(0, round(gap / med) - 1),
                })
        return out

    def _intervals(self) -> list[int]:
        return [self._ts[i + 1] - self._ts[i] for i in range(len(self._ts) - 1)]

    # ---- access / seek / iterate ----------------------------------------
    def seek(self, t_us: int) -> int:
        """Nearest-frame index for an absolute timestamp."""
        return self._src.index_at_time(t_us)

    def at(self, index: int) -> TimelineFrame:
        if index < 0:
            index += len(self._src)
        fs = self._src.frame(index)
        return TimelineFrame(
            index=index,
            frame_index=fs.frame_index,
            t_capture_us=fs.t_capture_us,
            t_rel_us=int(fs.t_capture_us - self.t0_us),
            fieldset=fs,
        )

    def window(
        self,
        start: int = 0,
        stop: int | None = None,
        step: int = 1,
    ) -> Iterator[TimelineFrame]:
        """Iterate a half-open [start, stop) index window with a stride. Default
        is the whole timeline. Stride > 1 is a decimation knob for export."""
        n = len(self._src)
        if stop is None or stop > n:
            stop = n
        if step < 1:
            raise ValueError("step must be >= 1")
        for i in range(start, stop, step):
            yield self.at(i)

    def __iter__(self) -> Iterator[TimelineFrame]:
        return self.window()
