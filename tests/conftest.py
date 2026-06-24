"""Shared fixtures. The synth_raw_parquet helper writes a producer-identical
raw.parquet for ANY descriptor using the shared column convention - so the test
suite is self-contained (no SentrixSim / SentrixCapture import) yet exercises the
exact at-rest contract the real producers emit.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from sentrix_contracts import (
    Descriptor,
    bundled_descriptor_path,
    column_for,
    dyn_columns,
    load_descriptor,
    mag_columns,
)

DESC_DIR = Path(__file__).parent / "descriptors"


def _descriptor_cases() -> list[tuple[str, Path]]:
    cases = [
        ("Mark2_tiny", DESC_DIR / "Mark2_tiny.json"),
        ("RobotFinger_v1", DESC_DIR / "RobotFinger_v1.json"),
    ]
    try:  # Mark2_v1 is canonical/bundled in sentrix_contracts when available
        cases.insert(0, ("Mark2_v1", Path(bundled_descriptor_path("Mark2_v1"))))
    except FileNotFoundError:
        pass
    return cases


DESCRIPTOR_CASES = _descriptor_cases()


@pytest.fixture(params=DESCRIPTOR_CASES, ids=[c[0] for c in DESCRIPTOR_CASES])
def descriptor_case(request) -> tuple[str, Descriptor]:
    name, path = request.param
    return name, load_descriptor(path)


def synth_raw_parquet(desc: Descriptor, out: Path, n_frames: int = 4) -> Path:
    """Producer-identical raw.parquet: sensor_id-keyed columns + self-describing
    metadata. Magnetic channels get a deterministic ramp; dynamics left NaN
    (matches the single-glove record path where dyn is joined separately)."""
    mag = mag_columns(desc)
    dyn = dyn_columns(desc)
    valid = [f"{s.sensor_id}.valid" for s in desc.sensors.values()]
    cols: dict[str, list] = {c: [] for c in (["t_capture_us", "frame_index", "trigger_id"]
                                             + mag + dyn + valid
                                             + ["sat_any", "dropout_any",
                                                "calib_version", "descriptor_version"])}
    for i in range(n_frames):
        cols["t_capture_us"].append(1000 * i)
        cols["frame_index"].append(i)
        cols["trigger_id"].append(i)
        cols["sat_any"].append(False)
        cols["dropout_any"].append(False)
        cols["calib_version"].append(0)
        cols["descriptor_version"].append(desc.descriptor_version)
        for s in desc.sensors.values():
            if s.modality == "magnetic":
                base = abs(hash(s.sensor_id)) % 50
                cols[column_for(s.sensor_id, "bx")].append(float(base + i))
                cols[column_for(s.sensor_id, "by")].append(float(base - i))
                cols[column_for(s.sensor_id, "bz")].append(float(base + 2 * i))
                if "temp" in s.channels:
                    cols[column_for(s.sensor_id, "temp")].append(25.0)
                cols[f"{s.sensor_id}.valid"].append(True)
            elif s.modality == "dynamics":
                for ch in ("ax", "ay", "az"):
                    cols[column_for(s.sensor_id, ch)].append(float("nan"))
                if "temp" in s.channels:
                    cols[f"dyn.{s.sensor_id}.temp_c"].append(float("nan"))
                cols[f"{s.sensor_id}.valid"].append(True)

    table = pa.table(cols)
    table = table.replace_schema_metadata({
        b"sentrix_capture_meta": json.dumps({
            "tool": "tests.synth", "version": "0.1.0",
            "descriptor_version": desc.descriptor_version,
            "descriptor_hash": desc.descriptor_hash or "",
        }).encode(),
        b"sentrix_descriptor_version": desc.descriptor_version.encode(),
        b"sentrix_descriptor_hash": (desc.descriptor_hash or "").encode(),
        b"ts_column": b"t_capture_us",
    })
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out)
    return out


@pytest.fixture
def raw_parquet(descriptor_case, tmp_path) -> tuple[str, Descriptor, Path]:
    name, desc = descriptor_case
    p = synth_raw_parquet(desc, tmp_path / f"{name}_raw.parquet")
    return name, desc, p
