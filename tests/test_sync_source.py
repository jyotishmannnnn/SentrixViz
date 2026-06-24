"""VIZ-P4 — SyncFrameSource over a Silver aligned table.

Self-contained: builds a Silver-format parquet with pyarrow (no SentrixSync /
SentrixDataEngine import), matching the exact at-rest contract DataEngine writes
(t_ref_us, frame_index, <stream>.cNNN, <stream>.valid, <stream>.confidence + KV
metadata), then exercises SyncFrameSource — including a dropout frame.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from sentrix_contracts import bundled_descriptor_path, load_descriptor
from sentrixviz import RenderModel
from sentrixviz.sources import SyncFrameSource, resolve_descriptor

N = 5
DROPOUT = 2  # frame index with a gap


def _write_silver(path: Path) -> tuple[str, str]:
    desc = load_descriptor(bundled_descriptor_path("Mark2_v1"))
    mag_ids = desc.ids("magnetic")
    n_sensor = len(mag_ids)            # 21 for Mark2_v1
    n_axis = 3
    width = n_sensor * n_axis
    cols: dict[str, pa.Array] = {
        "t_ref_us": pa.array(np.arange(N, dtype=np.int64) * 625),
        "frame_index": pa.array(np.arange(N, dtype=np.int64)),
    }
    for j in range(width):
        sensor_i, axis_i = divmod(j, n_axis)
        vals = np.full(N, float(sensor_i * 10 + axis_i), dtype=np.float32)
        vals[DROPOUT] = np.nan          # gap -> NaN, never fabricated
        cols[f"tactile_field.c{j:03d}"] = pa.array(vals)
    valid = np.ones(N, dtype=bool)
    valid[DROPOUT] = False
    cols["tactile_field.valid"] = pa.array(valid)
    cols["tactile_field.confidence"] = pa.array(
        np.where(valid, 1.0, 0.0).astype(np.float32))

    table = pa.table(cols).replace_schema_metadata({
        b"sentrixdataengine_schema_version": b"1.0",
        b"session_id": b"01J9SYNTH0000",
        b"reference_clock_id": b"glove_L_hub",
        b"grid_rate_hz": b"1600.0",
        b"streams": json.dumps({"tactile_field": {
            "key": "glove_L::tactile_field", "device_id": "glove_L",
            "payload_kind": "bmm350_cluster_uT", "units": "uT", "kernel": "continuous",
            "shape": [n_sensor, n_axis], "width": width, "coverage": 0.8}}).encode(),
        b"topology": json.dumps([{
            "device_id": "glove_L", "topology_ref": "Mark2_v1",
            "topology_hash": desc.descriptor_hash}]).encode(),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return mag_ids[0], desc.descriptor_hash


@pytest.fixture
def silver(tmp_path) -> tuple[Path, str, str]:
    p = tmp_path / "silver" / "aligned" / "part-000.parquet"
    sid0, dhash = _write_silver(p)
    return p, sid0, dhash


def test_basic_surface(silver):
    p, sid0, dhash = silver
    src = SyncFrameSource(p)
    assert len(src) == N
    assert src.ts_column == "t_ref_us"
    assert src.descriptor_version == "Mark2_v1"
    assert src.descriptor_hash == dhash
    assert src.reference_clock_id == "glove_L_hub"
    assert src.channels() == ["bx", "by", "bz"]
    assert src.timestamps()[:3] == [0, 625, 1250]


def test_frame_values_keyed_by_sensor_id(silver):
    p, sid0, _ = silver
    src = SyncFrameSource(p)
    fs = src.frame(0)
    assert len(fs.values) == 21
    # sensor 0, axes map to bx/by/bz with values 0,1,2
    assert fs.values[sid0] == {"bx": 0.0, "by": 1.0, "bz": 2.0}
    assert fs.valid[sid0] is True
    assert math.isclose(fs.magnitude(sid0), math.sqrt(0 + 1 + 4))


def test_dropout_frame_is_nan_and_invalid(silver):
    p, sid0, _ = silver
    src = SyncFrameSource(p)
    fs = src.frame(DROPOUT)
    assert fs.valid[sid0] is False
    assert all(math.isnan(v) for v in fs.values[sid0].values())  # never fabricated


def test_integrates_with_resolve_descriptor_and_model(silver):
    p, _, _ = silver
    src = SyncFrameSource(p)
    desc = resolve_descriptor(src)            # hash-checked against the artifact
    model = RenderModel.from_descriptor(desc)
    assert model is not None
    # index_at_time seek primitive
    assert src.index_at_time(625) == 1


def test_explicit_descriptor_override(silver, tmp_path):
    p, _, _ = silver
    desc_path = bundled_descriptor_path("Mark2_v1")
    src = SyncFrameSource(p, descriptor_path=desc_path)
    assert len(src.frame(0).values) == 21
