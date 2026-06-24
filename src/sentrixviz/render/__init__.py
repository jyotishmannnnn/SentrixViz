"""Renderers - the only components that produce pixels.

Phase 1: MatplotlibRenderer (headless PNG), the minimal viable renderer.
Phase 6 (noted, not built): a WebGL frontend implementing the same Renderer
protocol; nothing upstream of this package changes when it lands.
"""
from .mpl import MatplotlibRenderer

__all__ = ["MatplotlibRenderer"]
