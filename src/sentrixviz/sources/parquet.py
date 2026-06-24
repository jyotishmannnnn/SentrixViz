"""ParquetFrameSource - playback from a raw.parquet artifact.

Self-describing, exactly like SentrixDataEngine's resolver and SentrixSync's
adapter: the timestamp column and descriptor identity are read from the parquet
schema metadata written by the producer (SentrixCapture / SentrixSim), and
payload columns are discovered via sentrix_contracts.parse_column. The SAME
artifact a producer wrote is replayed here with no producer import.

What this must NEVER do:
  - import SentrixSim / SentrixCapture / Sync / DataEngine,
  - assume a column set, a sensor count, or a fixed ts column name,
  - interpolate across an invalid frame.
"""
from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from ..core.fields import FieldSet, bind_row
from sentrix_contracts import parse_column

_TS_FALLBACKS = ("t_capture_us", "t_master_us")


class ParquetFrameSource:
    """Read-only playback view over one raw.parquet file."""

    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._table = pq.read_table(self._path)
        meta = self._table.schema.metadata or {}
        self._meta = {k.decode(): v.decode() for k, v in meta.items()}
        ts = self._meta.get("ts_column")
        if ts is None:
            ts = next((c for c in _TS_FALLBACKS if c in self._table.column_names), None)
        if ts is None:
            raise ValueError(
                f"{self._path}: no ts_column in metadata and no known fallback "
                f"in columns {self._table.column_names[:6]}..."
            )
        self._ts_column = ts
        # Eager column extraction; raw.parquet for one session is small.
        self._cols = {c: self._table.column(c).to_pylist() for c in self._table.column_names}
        self._n = self._table.num_rows
        self._payload_cols = [c for c in self._table.column_names if parse_column(c) is not None]
        self._valid_cols = [c for c in self._table.column_names if c.endswith(".valid")]

    # ---- FrameSource protocol ------------------------------------------
    @property
    def descriptor_version(self) -> str:
        return self._meta.get("sentrix_descriptor_version", "")

    @property
    def descriptor_hash(self) -> str | None:
        return self._meta.get("sentrix_descriptor_hash") or None

    @property
    def ts_column(self) -> str:
        return self._ts_column

    def channels(self) -> list[str]:
        chans: set[str] = set()
        for c in self._payload_cols:
            parsed = parse_column(c)
            if parsed:
                chans.add(parsed[1])
        return sorted(chans)

    def timestamps(self) -> list[int]:
        return [int(t) for t in self._cols[self._ts_column]]

    def __len__(self) -> int:
        return self._n

    def frame(self, index: int) -> FieldSet:
        if index < 0:
            index += self._n
        if not 0 <= index < self._n:
            raise IndexError(index)
        row = {c: self._cols[c][index] for c in self._payload_cols + self._valid_cols}
        ts = self._cols[self._ts_column][index]
        fidx = self._cols.get("frame_index", list(range(self._n)))[index]
        return bind_row(row, t_capture_us=int(ts), frame_index=int(fidx))

    # ---- convenience ----------------------------------------------------
    def index_at_time(self, t_us: int) -> int:
        """Nearest-frame index for a timestamp (step/seek primitive)."""
        ts = self.timestamps()
        return min(range(len(ts)), key=lambda i: abs(ts[i] - t_us))

    @property
    def path(self) -> Path:
        return self._path
