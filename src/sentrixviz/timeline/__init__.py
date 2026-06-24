"""Timeline / sequence playback (Phase 2.5).

Moves the SDK from one descriptor+frame -> one PNG, to descriptor -> Timeline
-> ordered PNG sequence (-> optional video), reusing the frozen contracts:
FrameSource feeds frames, Timeline adds playback semantics, the existing Layers
and Renderer turn each frame into pixels. No new contract; no rendering here.
"""
from .sequence import render_sequence
from .timeline import Timeline, TimelineFrame
from .video import render_video

__all__ = ["Timeline", "TimelineFrame", "render_sequence", "render_video"]
