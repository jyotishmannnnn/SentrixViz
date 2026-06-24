"""Phase 2 end-to-end: every --view renders valid PNG bytes for every descriptor,
through the unchanged Renderer, with no per-hardware code path.
"""
from __future__ import annotations

import pytest

from sentrixviz.cli import _layers_for_view, build_parser, main
from sentrixviz.core.model import RenderModel
from sentrixviz.render import MatplotlibRenderer
from sentrixviz.sources import ParquetFrameSource

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_VIEWS = ["raw", "heatmap", "clusters", "centroid", "shear"]


@pytest.mark.parametrize("view", _VIEWS)
def test_every_view_renders_png(raw_parquet, view):
    name, desc, path = raw_parquet
    model = RenderModel.from_descriptor(desc)
    fs = ParquetFrameSource(path).frame(0)
    layers = _layers_for_view(view, channel="mag", modality=None)
    png = MatplotlibRenderer().render(model, layers, fieldset=fs)
    assert isinstance(png, bytes) and png.startswith(_PNG_MAGIC)


def test_cli_render_view_writes_file(raw_parquet, tmp_path):
    """CLI path for a Phase-2 view, exercised for every descriptor."""
    name, desc, path = raw_parquet
    out = tmp_path / "heatmap.png"
    # tests/descriptors holds the JSON for the synthetic revisions; Mark2_v1 is
    # bundled and resolves from the artifact metadata.
    desc_arg = ["--descriptor", str(_descriptor_path(name))] if name != "Mark2_v1" else []
    rc = main(["render", str(path), *desc_arg, "--view", "heatmap", "-o", str(out)])
    assert rc == 0 and out.exists() and out.read_bytes().startswith(_PNG_MAGIC)


def test_default_view_is_backward_compatible():
    """render still defaults to raw (Phase-1 behaviour)."""
    ns = build_parser().parse_args(["render", "x.parquet"])
    assert ns.view == "raw"


def _descriptor_path(name: str):
    from pathlib import Path
    return Path(__file__).parent / "descriptors" / f"{name}.json"
