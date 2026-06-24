"""sentrixviz.core - the pure SDK boundary.

Depends ONLY on sentrix_contracts + numpy. No I/O, no rendering backend, no UI.
Everything here is testable without a GPU or a file.
"""
from .fields import FieldSet, bind_row
from .model import ClusterView, Node, RenderModel
from .protocols import FrameSource, Layer, Renderer

__all__ = [
    "RenderModel",
    "Node",
    "ClusterView",
    "FieldSet",
    "bind_row",
    "FrameSource",
    "Layer",
    "Renderer",
]
