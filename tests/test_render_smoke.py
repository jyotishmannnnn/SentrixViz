"""Proves the full boundary end-to-end: descriptor -> model -> layer -> PNG, for
every descriptor, with NO code change between them. Asserts valid PNG bytes.
"""
from __future__ import annotations

from sentrixviz.core.model import RenderModel
from sentrixviz.layers import FieldLayer, TopologyLayer
from sentrixviz.render import MatplotlibRenderer
from sentrixviz.sources import ParquetFrameSource

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_topology_render_png(descriptor_case):
    name, desc = descriptor_case
    model = RenderModel.from_descriptor(desc)
    png = MatplotlibRenderer().render(model, [TopologyLayer()], fieldset=None)
    assert isinstance(png, bytes) and png.startswith(_PNG_MAGIC)


def test_field_render_png(raw_parquet):
    name, desc, path = raw_parquet
    model = RenderModel.from_descriptor(desc)
    fs = ParquetFrameSource(path).frame(0)
    png = MatplotlibRenderer().render(model, [FieldLayer(reduction="mag")], fieldset=fs)
    assert isinstance(png, bytes) and png.startswith(_PNG_MAGIC)


def test_field_render_to_file(raw_parquet, tmp_path):
    name, desc, path = raw_parquet
    model = RenderModel.from_descriptor(desc)
    fs = ParquetFrameSource(path).frame(0)
    out = tmp_path / "frame.png"
    res = MatplotlibRenderer().render(model, [FieldLayer("bz", modality="magnetic")],
                                      fieldset=fs, out=out)
    assert res == out and out.exists() and out.read_bytes().startswith(_PNG_MAGIC)
