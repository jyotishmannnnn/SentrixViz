"""LiveSession - the Phase 3 orchestration loop.

The live analogue of render_sequence: pull FieldSets off a LiveFrameSource and
push each through the SAME Renderer.render(model, layers, fieldset) path a
single frame uses. It owns NO rendering and introduces NO contract - it is the
unbounded, latest-wins sibling of the (finite, windowed) Timeline orchestration.

Differences from render_sequence, all forced by "live" not "playback":
  - the stream is unbounded -> stop on max_frames or feed end, never a scan;
  - normalization cannot scan ahead -> RollingNormalizer (bounded, adaptive),
    primed once from the first frame for the facts that are frame-constant
    (vector anchor span, cluster label set);
  - output is a latest-wins monitor (`out`, overwritten each frame) and/or a
    numbered sequence + manifest (`out_dir`) - the manifest carries t_capture_us
    per frame, the video-overlay / sync key, identical to render_sequence.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from ..core.model import RenderModel
from ..core.protocols import Layer, Renderer
from ..normalize import RollingNormalizer
from ..sources.live import LiveFrameSource


def _prime_static_facts(policy: RollingNormalizer, model: RenderModel,
                        layers: Sequence[Layer], fieldset) -> None:
    """Seed the frame-INDEPENDENT facts a rolling policy cannot learn from a
    single per-call value: the anchor-cloud span per vector key (positions never
    move) and the full cluster label set (membership is descriptor-fixed). Done
    once on the first frame; keeps shear arrows and cluster colours stable."""
    for layer in layers:
        prims = layer.encode(model, fieldset)
        vecs = prims.get("vectors")
        if vecs is not None and vecs.get("key") is not None:
            xy = np.asarray(vecs.get("xy"), dtype=float)
            if xy.size:
                span = float(np.nanmax(xy.max(axis=0) - xy.min(axis=0)))
                policy.prime_vector_span(vecs["key"], span)
        pts = prims.get("points")
        if pts is not None and pts.get("kind") == "category" and pts.get("key"):
            policy.observe_categories(pts["key"], list(pts.get("category_labels", [])))


class LiveSession:
    """Render an unbounded stream of FieldSets through the existing stack."""

    def __init__(
        self,
        model: RenderModel,
        layers: Sequence[Layer],
        renderer: Renderer,
        *,
        normalizer: RollingNormalizer | None = None,
        view: str | None = None,
        channel: str | None = None,
        name_width: int = 6,
    ):
        self._model = model
        self._layers = layers
        self._renderer = renderer
        self._policy = normalizer or RollingNormalizer()
        self._view = view
        self._channel = channel
        self._name_width = name_width

    def run(
        self,
        live: LiveFrameSource,
        *,
        out: str | Path | None = None,
        out_dir: str | Path | None = None,
        max_frames: int | None = None,
        fps: float | None = None,
        on_frame: Callable[[dict], None] | None = None,
        write_manifest: bool = True,
    ) -> dict:
        """Consume `live` until it ends or `max_frames` frames are rendered.

        `out`     - a single PNG overwritten every frame (latest-wins monitor).
        `out_dir` - numbered PNGs (000000.png, ...) + manifest.json.
        At least one of `out` / `out_dir` should be given. `fps` throttles
        playback (None = as fast as frames arrive). `on_frame(record)` is called
        after each rendered frame (a UI / progress hook).
        """
        out = Path(out) if out is not None else None
        out_dir = Path(out_dir) if out_dir is not None else None
        if out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)

        # apply the rolling policy for the duration of the session, then restore
        prev = getattr(self._renderer, "normalizer", None)
        if hasattr(self._renderer, "set_normalizer"):
            self._renderer.set_normalizer(self._policy)

        records: list[dict] = []
        primed = False
        try:
            for emit, fs in enumerate(live):
                if not primed:
                    _prime_static_facts(self._policy, self._model, self._layers, fs)
                    primed = True

                if out_dir is not None:
                    fname = f"{emit:0{self._name_width}d}.png"
                    self._renderer.render(self._model, self._layers,
                                          fieldset=fs, out=out_dir / fname)
                    file_ref = fname
                else:
                    file_ref = None

                if out is not None:
                    # latest-wins monitor; written even when out_dir is also set
                    self._renderer.render(self._model, self._layers,
                                          fieldset=fs, out=out)
                    file_ref = file_ref or out.name

                rec = {
                    "file": file_ref,
                    "emit": emit,
                    "frame_index": fs.frame_index,
                    "t_capture_us": fs.t_capture_us,
                }
                records.append(rec)
                if on_frame is not None:
                    on_frame(rec)

                if max_frames is not None and emit + 1 >= max_frames:
                    break
                if fps and fps > 0:
                    import time
                    time.sleep(1.0 / fps)
        finally:
            if hasattr(self._renderer, "set_normalizer") and prev is not None:
                self._renderer.set_normalizer(prev)

        manifest = {
            "descriptor_version": self._model.descriptor_version,
            "descriptor_hash": self._model.descriptor_hash,
            "device_class": self._model.device_class,
            "mode": "live",
            "view": self._view,
            "channel": self._channel,
            "n_frames": len(records),
            "name_width": self._name_width,
            "dropped": getattr(getattr(live, "_feed", None), "dropped", 0),
            "frames": records,
        }
        if out_dir is not None and write_manifest:
            (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return manifest
