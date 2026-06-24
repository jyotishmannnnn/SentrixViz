"""render_sequence - descriptor -> Timeline -> existing layers/renderer -> an
ordered PNG sequence + a JSON manifest.

This is pure orchestration: it owns NO rendering. Every frame goes through the
SAME Renderer.render(model, layers, fieldset) path a single-frame render uses,
so there is zero rendering duplication and `--view raw|heatmap|clusters|
centroid|shear` behave frame-for-frame identically to `sentrixviz render`.

The manifest is the forward-compatibility hook: it records each frame's
absolute t_capture_us alongside its file, which is exactly what a future
camera/video synchronization or overlay tool needs to align streams by time.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from ..core.model import RenderModel
from ..core.protocols import Layer, Renderer
from ..normalize import NormalizationPolicy, scan_ranges
from .timeline import Timeline


def _frame_name(index: int, width: int) -> str:
    return f"{index:0{width}d}.png"


def _render_window(renderer, model, layers, window, out_dir: Path, name_width: int) -> list[dict]:
    """Render each materialized frame through the SAME single-frame render path,
    returning the per-frame manifest records in emission order."""
    frames: list[dict] = []
    for emit, tf in enumerate(window):
        fname = _frame_name(emit, name_width)
        renderer.render(model, layers, fieldset=tf.fieldset, out=out_dir / fname)
        frames.append({
            "file": fname,
            "index": tf.index,             # source index
            "frame_index": tf.frame_index, # producer counter
            "t_capture_us": tf.t_capture_us,
            "t_rel_us": tf.t_rel_us,
        })
    return frames


def render_sequence(
    model: RenderModel,
    layers: Sequence[Layer],
    timeline: Timeline,
    out_dir: str | Path,
    renderer: Renderer,
    *,
    start: int = 0,
    stop: int | None = None,
    step: int = 1,
    name_width: int = 6,
    view: str | None = None,
    channel: str | None = None,
    write_manifest: bool = True,
    normalize: NormalizationPolicy | str | None = "global",
) -> dict:
    """Render a contiguous (or strided) window of the timeline to numbered PNGs.

    Output files are 000000.png, 000001.png, ... in emission order (NOT the
    producer frame_index, so the sequence is dense even across drops). Returns
    the manifest dict (also written to manifest.json unless disabled).

    `normalize` controls colour/size/length stability across the window:
      "global" (default) - scan the window once, fix every range so nothing
                           flickers during playback;
      None / "perframe"  - leave the renderer's own policy (legacy per-frame);
      a NormalizationPolicy instance - use it verbatim.
    The renderer's previous policy is restored on exit, so passing one in is a
    non-destructive, scoped swap.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Materialize the window once: it feeds both the range scan and the render.
    window = list(timeline.window(start=start, stop=stop, step=step))

    policy: NormalizationPolicy | None
    if normalize == "global":
        policy = scan_ranges(model, layers, [tf.fieldset for tf in window])
    elif normalize in (None, "perframe"):
        policy = None
    elif isinstance(normalize, str):
        raise ValueError(f"unknown normalize mode: {normalize!r}")
    else:
        policy = normalize

    prev_policy = getattr(renderer, "normalizer", None)
    if policy is not None and hasattr(renderer, "set_normalizer"):
        renderer.set_normalizer(policy)
    try:
        frames = _render_window(renderer, model, layers, window, out_dir, name_width)
    finally:
        if policy is not None and prev_policy is not None and hasattr(renderer, "set_normalizer"):
            renderer.set_normalizer(prev_policy)

    manifest = {
        "descriptor_version": model.descriptor_version,
        "descriptor_hash": model.descriptor_hash,
        "device_class": model.device_class,
        "view": view,
        "channel": channel,
        "n_frames": len(frames),
        "t0_us": timeline.t0_us,
        "duration_us": timeline.duration_us,
        "fps_estimate": round(timeline.fps_estimate, 4),
        "dropped": timeline.dropped(),
        "name_width": name_width,
        "frames": frames,
    }
    if write_manifest:
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
