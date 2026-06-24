"""Tier-3 cluster layers: ClusterLayer / CentroidLayer / ShearLayer.

All three read per-cluster features from `derived.cluster_features` (membership
from the descriptor only) and emit renderer-agnostic `markers` / `vectors`. They
never parse sensor names, never assume a count, and never choose a colour.
"""
from __future__ import annotations

import numpy as np

from ..core.fields import FieldSet
from ..core.model import RenderModel
from .derived import cluster_features


def _stack(feats, attr) -> np.ndarray:
    return np.asarray([getattr(f, attr) for f in feats], dtype=float) if feats else np.zeros((0, 2))


class ClusterLayer:
    """One glyph per descriptor cluster at its geometric centre:
    colour = normal_proxy (activation), size = activation, arrow = shear."""
    def __init__(self, reduction: str = "mag"):
        self.reduction = reduction
        self.name = "clusters"

    def encode(self, model: RenderModel, fieldset: FieldSet | None) -> dict:
        if fieldset is None:
            raise ValueError("ClusterLayer requires a fieldset")
        feats = cluster_features(model, fieldset, self.reduction)
        centers = _stack(feats, "center_xy")
        normal = np.asarray([f.normal_proxy for f in feats], dtype=float)
        activation = np.asarray([f.activation for f in feats], dtype=float)
        shear = _stack(feats, "shear_xy")
        return {
            "title": f"{model.descriptor_version} - clusters "
                     f"({len(feats)}) @ frame={fieldset.frame_index}",
            "markers": {"key": "markers:clusters", "xy": centers, "style": "star",
                        "weight": activation, "values": normal},
            "vectors": {"key": "vectors:clusters", "xy": centers, "uv": shear,
                        "label": "shear"},
        }


class CentroidLayer:
    """Response-weighted contact centroid per cluster, sized by activation."""
    def __init__(self, reduction: str = "mag"):
        self.reduction = reduction
        self.name = "centroid"

    def encode(self, model: RenderModel, fieldset: FieldSet | None) -> dict:
        if fieldset is None:
            raise ValueError("CentroidLayer requires a fieldset")
        feats = cluster_features(model, fieldset, self.reduction)
        centroids = _stack(feats, "centroid_xy")
        activation = np.asarray([f.activation for f in feats], dtype=float)
        return {
            "title": f"{model.descriptor_version} - centroid "
                     f"({len(feats)} clusters) @ frame={fieldset.frame_index}",
            "markers": {"key": "markers:centroid", "xy": centroids, "style": "star",
                        "weight": activation, "values": activation},
        }


class ShearLayer:
    """Per-cluster lateral-field shear vectors, anchored at cluster centres."""
    def __init__(self, reduction: str = "mag"):
        self.reduction = reduction
        self.name = "shear"

    def encode(self, model: RenderModel, fieldset: FieldSet | None) -> dict:
        if fieldset is None:
            raise ValueError("ShearLayer requires a fieldset")
        feats = cluster_features(model, fieldset, self.reduction)
        centers = _stack(feats, "center_xy")
        shear = _stack(feats, "shear_xy")
        mag = np.asarray([f.shear_mag for f in feats], dtype=float)
        return {
            "title": f"{model.descriptor_version} - shear "
                     f"({len(feats)} clusters) @ frame={fieldset.frame_index}",
            "markers": {"key": "markers:shear", "xy": centers, "style": "dot",
                        "weight": mag, "values": None},
            "vectors": {"key": "vectors:shear", "xy": centers, "uv": shear,
                        "label": "shear"},
        }
