"""SyncFrameSource (VIZ-P4) — playback over SentrixDataEngine's Silver table.

Phase 1's ParquetFrameSource reads PRODUCER RAW (`mag.<sid>.*` columns, producer
clock). This source reads the SYNCED / aligned Silver artifact DataEngine writes
from a SyncResult:

    t_ref_us, frame_index, <stream>.cNNN (flattened payload), <stream>.valid,
    <stream>.confidence    + KV metadata {streams, topology, reference_clock_id}

So Viz can finally render the *aligned, dropout-aware* pipeline result on the
reference timeline — not just producer raw. It is a drop-in `FrameSource`: it
exposes descriptor identity, so the existing `resolve_descriptor` →
`RenderModel.from_descriptor` → layers → renderer path is unchanged.

The Silver columns are generic `<stream>.cNNN` (a flattened `[n_sensor, n_axis]`
payload), so this source inverts them back to `values[sensor_id][channel]` using
the stream's `shape`/`payload_kind` (from the file's KV metadata) and the
descriptor's per-modality sensor order. Topology-driven: counts/order come from
the descriptor, never hardcoded.

What this must NEVER do:
  - import SentrixSync / SentrixDataEngine (it reads their on-disk artifact),
  - fabricate values across a gap (invalid grid points stay NaN, valid=False),
  - hold positions/topology (that is RenderModel).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from sentrix_contracts import Descriptor, bundled_descriptor_path, load_descriptor

from ..core.fields import FieldSet

# Per payload_kind: (descriptor modality, ordered channel names per axis). This
# is the same channel convention as sentrix_contracts.columns (bx/by/bz, ax/ay/az);
# the SET of sensors/columns still comes from the descriptor at runtime.
_KIND = {
    "bmm350_cluster_uT": ("magnetic", ("bx", "by", "bz")),
    "lis2dtw12_accel_g": ("dynamics", ("ax", "ay", "az")),
}

_TS_COLUMN = "t_ref_us"


@dataclass(frozen=True)
class _ColMap:
    """One Silver value column → (sensor_id, channel)."""
    column: str
    sensor_id: str
    channel: str


class SyncFrameSource:
    """Read-only playback over one Silver `aligned/part-000.parquet`."""

    def __init__(self, path: str | Path, *,
                 descriptor: Descriptor | None = None,
                 descriptor_path: str | Path | None = None):
        self._path = Path(path)
        self._table = pq.read_table(self._path)
        meta = self._table.schema.metadata or {}
        self._meta = {k.decode(): v.decode() for k, v in meta.items()}
        if _TS_COLUMN not in self._table.column_names:
            raise ValueError(
                f"{self._path}: not a Silver aligned table (no {_TS_COLUMN!r} column)")

        self._streams = json.loads(self._meta.get("streams", "{}"))
        self._topology = json.loads(self._meta.get("topology", "[]"))
        self._topo_by_device = {t.get("device_id"): t for t in self._topology}

        # Descriptor resolution (explicit override, else bundled by topology_ref).
        self._explicit_desc = descriptor
        if descriptor is None and descriptor_path is not None:
            self._explicit_desc = load_descriptor(descriptor_path)
        self._desc_cache: dict[str, Descriptor] = {}

        self._colmaps: list[_ColMap] = []
        self._valid_col_for_sensor: dict[str, str] = {}
        self._build_mapping()

        self._n = self._table.num_rows
        valid_cols = sorted(set(self._valid_col_for_sensor.values()))
        wanted = [_TS_COLUMN, "frame_index"] \
            + [c.column for c in self._colmaps] + valid_cols
        cols_present = [c for c in wanted if c in self._table.column_names]
        self._cols = {c: self._table.column(c).to_pylist() for c in cols_present}

    # ---- mapping construction ------------------------------------------
    def _descriptor_for(self, device_id: str) -> Descriptor:
        if self._explicit_desc is not None:
            return self._explicit_desc
        if device_id in self._desc_cache:
            return self._desc_cache[device_id]
        topo = self._topo_by_device.get(device_id) or {}
        ref = topo.get("topology_ref")
        if not ref:
            raise ValueError(
                f"{self._path}: no topology_ref for device {device_id!r} in Silver "
                "metadata; pass descriptor= or descriptor_path= explicitly")
        desc = load_descriptor(bundled_descriptor_path(ref))
        want = topo.get("topology_hash")
        if want and desc.descriptor_hash and want != desc.descriptor_hash:
            raise ValueError(
                f"descriptor hash mismatch for {device_id!r}: "
                f"artifact={want} descriptor={desc.descriptor_hash}")
        self._desc_cache[device_id] = desc
        return desc

    def _build_mapping(self) -> None:
        for name, sm in self._streams.items():
            payload_kind = sm.get("payload_kind", "")
            shape = tuple(sm.get("shape") or ())
            if payload_kind not in _KIND:
                raise ValueError(
                    f"{self._path}: stream {name!r} payload_kind {payload_kind!r} "
                    f"not renderable; known kinds: {sorted(_KIND)}")
            if len(shape) < 2:
                raise ValueError(
                    f"{self._path}: stream {name!r} needs a [n_sensor, n_axis] shape "
                    f"to map columns, got {shape}")
            modality, axes = _KIND[payload_kind]
            n_sensor, n_axis = shape[0], shape[-1]
            ids = self._descriptor_for(sm.get("device_id")).ids(modality)
            if len(ids) < n_sensor:
                raise ValueError(
                    f"{self._path}: descriptor has {len(ids)} {modality} sensors but "
                    f"stream {name!r} declares {n_sensor}")
            width = int(sm.get("width") or (n_sensor * n_axis))
            valid_col = f"{name}.valid"
            for j in range(width):
                sensor_i, axis_i = divmod(j, n_axis)
                sid = ids[sensor_i]
                ch = axes[axis_i] if axis_i < len(axes) else f"c{axis_i}"
                self._colmaps.append(_ColMap(f"{name}.c{j:03d}", sid, ch))
                self._valid_col_for_sensor[sid] = valid_col

    # ---- FrameSource protocol ------------------------------------------
    @property
    def descriptor_version(self) -> str:
        for t in self._topology:
            if t.get("topology_ref"):
                return t["topology_ref"]
        return self._explicit_desc.descriptor_version if self._explicit_desc else ""

    @property
    def descriptor_hash(self) -> str | None:
        for t in self._topology:
            if t.get("topology_hash"):
                return t["topology_hash"]
        return self._explicit_desc.descriptor_hash if self._explicit_desc else None

    @property
    def ts_column(self) -> str:
        return _TS_COLUMN

    @property
    def reference_clock_id(self) -> str | None:
        return self._meta.get("reference_clock_id") or None

    def channels(self) -> list[str]:
        return sorted({c.channel for c in self._colmaps})

    def timestamps(self) -> list[int]:
        return [int(t) for t in self._cols[_TS_COLUMN]]

    def __len__(self) -> int:
        return self._n

    def frame(self, index: int) -> FieldSet:
        if index < 0:
            index += self._n
        if not 0 <= index < self._n:
            raise IndexError(index)
        values: dict[str, dict[str, float]] = {}
        for cm in self._colmaps:
            v = self._cols[cm.column][index]
            values.setdefault(cm.sensor_id, {})[cm.channel] = (
                float(v) if v is not None else float("nan"))
        valid: dict[str, bool] = {}
        for sid, vcol in self._valid_col_for_sensor.items():
            valid[sid] = bool(self._cols[vcol][index]) if vcol in self._cols else True
        ts = self._cols[_TS_COLUMN][index]
        fidx = self._cols.get("frame_index", list(range(self._n)))[index]
        return FieldSet(t_capture_us=int(ts), frame_index=int(fidx),
                        values=values, valid=valid)

    # ---- convenience ----------------------------------------------------
    def index_at_time(self, t_us: int) -> int:
        ts = self.timestamps()
        return min(range(len(ts)), key=lambda i: abs(ts[i] - t_us))

    @property
    def path(self) -> Path:
        return self._path
