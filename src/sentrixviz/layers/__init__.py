"""Layers - pure (model, fieldset) -> primitives encoders.

Primitives schema (renderer-agnostic dict):
  {
    "title": str,
    "points": {                 # optional
       "xy": np.ndarray (N,2),
       "ids": list[str],
       "kind": "scalar" | "category",
       "values": np.ndarray (N,) float,        # when kind == "scalar"
       "categories": np.ndarray (N,) int,      # when kind == "category"
       "category_labels": list[str],           # when kind == "category"
       "valid": np.ndarray (N,) bool,
    },
    "segments": {               # optional
       "xy0": np.ndarray (M,2), "xy1": np.ndarray (M,2),
    },
    "field": {                  # optional (Phase 2) - interpolated raster
       "grid": np.ndarray (R,R) float, NaN outside support,
       "extent": (x0, x1, y0, y1),
    },
    "vectors": {                # optional (Phase 2) - arrows (e.g. shear)
       "xy": np.ndarray (K,2), "uv": np.ndarray (K,2), "label": str | None,
    },
    "markers": {                # optional (Phase 2) - emphasis points
       "xy": np.ndarray (K,2),
       "weight": np.ndarray (K,) float | None,   # renderer maps -> size
       "values": np.ndarray (K,) float | None,   # renderer maps -> color
       "style": "dot" | "star",
    },
  }

A layer NEVER chooses colors, sizes, or touches a backend. It emits physical
scalars / vectors / category indices + geometry; the Renderer maps those to pixels.

Phase 1 ships TopologyLayer (static geometry) + FieldLayer (Tier 1 raw values).
Phase 2 adds the topology-aware layers: HeatmapLayer (Tier 2 continuous field)
and the cluster reductions ClusterLayer / CentroidLayer / ShearLayer (Tier 3).
"""
from .cluster import CentroidLayer, ClusterLayer, ShearLayer
from .field import FieldLayer
from .heatmap import HeatmapLayer
from .topology import TopologyLayer

__all__ = [
    "TopologyLayer",
    "FieldLayer",
    "HeatmapLayer",
    "ClusterLayer",
    "CentroidLayer",
    "ShearLayer",
]
