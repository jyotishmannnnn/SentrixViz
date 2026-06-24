"""FieldSet - one frame's sensor values, bound to sensor_ids.

L1 of the SDK: a parquet row (sensor_id-keyed columns) becomes a per-sensor,
per-channel value map. Binding reuses sentrix_contracts.parse_column verbatim,
so a layout change (different column SET) needs no code change here.

What this module must NEVER do:
  - assume which sensors / channels exist (read them from the row),
  - hold positions or topology (that is RenderModel),
  - fabricate values across a dropout (invalid -> NaN, valid flag preserved).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from sentrix_contracts import parse_column


@dataclass(frozen=True)
class FieldSet:
    """Values for one frame. `values[sid][channel]` -> float (NaN if absent)."""
    t_capture_us: int
    frame_index: int
    values: dict[str, dict[str, float]]
    valid: dict[str, bool]

    # ---- reductions (display-only; never persisted) ---------------------
    def channel(self, sensor_id: str, ch: str) -> float:
        return self.values.get(sensor_id, {}).get(ch, float("nan"))

    def magnitude(self, sensor_id: str) -> float:
        """Vector magnitude over whatever vector channels the sensor carries
        (bx/by/bz for magnetic, ax/ay/az for dynamics). Count-agnostic."""
        v = self.values.get(sensor_id, {})
        comps = [v[c] for c in ("bx", "by", "bz", "ax", "ay", "az") if c in v]
        if not comps:
            return float("nan")
        return math.sqrt(sum(x * x for x in comps))

    def scalar(self, sensor_id: str, reduction: str = "mag") -> float:
        """Resolve a scalar for rendering: a raw channel name, or 'mag'."""
        if reduction == "mag":
            return self.magnitude(sensor_id)
        return self.channel(sensor_id, reduction)


def bind_row(row: dict[str, object], t_capture_us: int, frame_index: int) -> FieldSet:
    """Project one parquet row (column -> value) into a FieldSet.

    Payload columns are recognized by parse_column; non-payload columns
    (timestamps, flags) are ignored. Validity is read from `<sid>.valid` when
    present, defaulting to True.
    """
    values: dict[str, dict[str, float]] = {}
    valid: dict[str, bool] = {}
    for col, val in row.items():
        parsed = parse_column(col)
        if parsed is not None:
            sid, ch = parsed
            values.setdefault(sid, {})[ch] = float(val) if val is not None else float("nan")
            continue
        if col.endswith(".valid"):
            valid[col[: -len(".valid")]] = bool(val)
    return FieldSet(
        t_capture_us=int(t_capture_us),
        frame_index=int(frame_index),
        values=values,
        valid=valid,
    )
