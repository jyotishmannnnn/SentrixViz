"""Proves playback: a producer-identical raw.parquet round-trips through
ParquetFrameSource -> FieldSet for every descriptor, self-describing (no
descriptor passed to the source), with provenance check on resolve.
"""
from __future__ import annotations

import math

from sentrixviz.core.fields import FieldSet
from sentrixviz.sources import ParquetFrameSource, resolve_descriptor


def test_source_self_describes(raw_parquet):
    name, desc, path = raw_parquet
    src = ParquetFrameSource(path)
    assert src.ts_column == "t_capture_us"
    assert src.descriptor_version == desc.descriptor_version
    assert len(src) == 4
    assert "bx" in src.channels()


def test_resolve_descriptor_checks_hash(raw_parquet):
    name, desc, path = raw_parquet
    src = ParquetFrameSource(path)
    resolved = resolve_descriptor(src, descriptor_path=None) if name == "Mark2_v1" \
        else resolve_descriptor(src, descriptor_path=_local(name))
    assert resolved.descriptor_version == desc.descriptor_version


def test_frame_binds_values(raw_parquet):
    name, desc, path = raw_parquet
    src = ParquetFrameSource(path)
    fs = src.frame(0)
    assert isinstance(fs, FieldSet)
    # every magnetic sensor has bx/by/bz bound and a finite magnitude
    for s in desc.sensors.values():
        if s.modality == "magnetic":
            assert set(("bx", "by", "bz")).issubset(fs.values[s.sensor_id])
            assert math.isfinite(fs.magnitude(s.sensor_id))


def _local(name: str):
    from pathlib import Path
    return Path(__file__).parent / "descriptors" / f"{name}.json"
