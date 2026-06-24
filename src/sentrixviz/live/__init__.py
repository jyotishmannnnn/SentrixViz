"""Live visualization orchestration (Phase 3).

Unbounded, latest-wins sibling of timeline/ playback. A LiveFrameSource (over a
FrameFeed transport) feeds FieldSets; LiveSession pushes each through the SAME
existing Layers + Renderer, with a RollingNormalizer for bounded, adaptive,
flicker-free range mapping. No new contract; live is just another FrameSource.
"""
from .session import LiveSession

__all__ = ["LiveSession"]
