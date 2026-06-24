"""SentrixViz - topology-driven visualization SDK for the Sentrix stack.

Posture (inherited from the ecosystem, non-negotiable):
  - Consumes the descriptor + artifacts; NEVER imports Sim/Sync/DataEngine/
    Capture as code (sits beside the pipeline, like Sync does toward producers).
  - Zero geometry constants: every spatial fact comes from a Descriptor,
    addressed by descriptor_hash. A new hardware revision is a new descriptor.
  - Live vs playback is a FrameSource distinction, not a rendering one.
  - Derived/display signals are never persisted as canonical; invalid frames are
    never interpolated.

Layered SDK: core (pure) -> sources (adapters) -> layers (encoders) ->
render (pixels). See ARCHITECTURE.md.
"""
from .core import FieldSet, FrameSource, Layer, RenderModel, Renderer, bind_row
from .normalize import (
    GlobalNormalizer,
    NormalizationPolicy,
    PerFrameNormalizer,
    RollingNormalizer,
    scan_ranges,
)

__all__ = [
    "RenderModel", "FieldSet", "FrameSource", "Renderer", "Layer", "bind_row",
    "NormalizationPolicy", "PerFrameNormalizer", "GlobalNormalizer",
    "RollingNormalizer", "scan_ranges",
]
__version__ = "0.1.0"
