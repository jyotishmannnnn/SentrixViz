"""Ephemeral per-cluster derived features for the Phase-2 cluster layers.

The canonical proxy formulas (normal_proxy / shear / centroid / activation) now
live in ONE place — `sentrix_contracts.derived` (the VIZ-P5 seam) — and are called
by BOTH this display path and SentrixDataEngine's batch `exporters/derived.py`, so
they can never drift. We still do not import DataEngine (hard rule #3) and never
persist these (hard rule #6); this module only binds the shared math to a
`FieldSet` + `RenderModel` and shapes the per-cluster feature records.

Single-frame simplification: the canonical baseline `B0[k] = median_t B` is a
per-sensor median over time. A `Layer` sees one `FieldSet` (one frame), so
Phase 2 takes `B0 = 0` and uses the raw per-sensor magnitude as the response
`r`. Documented here, not hidden.

Cluster membership comes ONLY from the descriptor (`RenderModel.clusters` /
`Node.cluster_id`). Sensor names are NEVER parsed. A descriptor with no clusters
falls back to one bucket per sensor - exactly the canonical exporter's fallback.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sentrix_contracts.derived import (
    activation as _activation,
    geometric_center,
    normal_proxy,
    response_weighted_centroid,
    shear_magnitude,
    shear_vector,
)

from ..core.fields import FieldSet
from ..core.model import RenderModel


@dataclass(frozen=True)
class ClusterFeature:
    cluster_id: str
    center_xy: np.ndarray      # (2,) geometric mean of member positions
    centroid_xy: np.ndarray    # (2,) response-weighted centroid; NaN if no activation
    normal_proxy: float        # mean member response magnitude  (common-mode)
    shear_xy: np.ndarray       # (2,) mean lateral field shift
    shear_mag: float
    activation: float          # sum of member responses (centroid weight)
    n_members: int


def cluster_membership(model: RenderModel) -> dict[str, list[str]]:
    """cluster_id -> positioned member ids, from the descriptor only.

    Falls back to one bucket per positioned sensor when the descriptor declares
    no clusters (matches the canonical exporter's per-sensor fallback)."""
    xy = model.project_2d()
    groups: dict[str, list[str]] = {}
    for c in model.clusters:
        members = [m for m in c.members if m in xy]
        if members:
            groups[c.id] = members
    if not groups:
        groups = {sid: [sid] for sid in model.positioned_ids()}
    return groups


def _lateral(fieldset: FieldSet, sid: str) -> tuple[float, float]:
    """First two vector components for shear: (bx,by) for magnetic, (ax,ay) for
    dynamics. Channel set is read from the frame, never assumed by modality."""
    v = fieldset.values.get(sid, {})
    if "bx" in v and "by" in v:
        return v["bx"], v["by"]
    if "ax" in v and "ay" in v:
        return v["ax"], v["ay"]
    return float("nan"), float("nan")


def cluster_features(
    model: RenderModel, fieldset: FieldSet, reduction: str = "mag"
) -> list[ClusterFeature]:
    """Per-cluster (normal_proxy, shear, centroid, activation) for one frame.

    Faithful to the canonical formulas with baseline B0 = 0 (single frame).
    Invalid members contribute NaN and are skipped by the nan-aware reductions.
    """
    xy = model.project_2d()
    out: list[ClusterFeature] = []
    for cid, members in cluster_membership(model).items():
        pos = np.asarray([xy[m] for m in members], dtype=float)          # (m, 2)
        r = np.asarray([fieldset.scalar(m, reduction) for m in members], dtype=float)
        valid = np.asarray([fieldset.valid.get(m, True) for m in members], dtype=bool)
        r = np.where(valid, r, np.nan)
        lat = np.asarray([_lateral(fieldset, m) for m in members], dtype=float)  # (m, 2)

        # Canonical cluster math shared with SentrixDataEngine (VIZ-P5 seam).
        center = geometric_center(pos)
        normal = float(normal_proxy(r))
        shear = np.asarray(shear_vector(lat), dtype=float)
        shear_mag = float(shear_magnitude(shear))
        wsum = float(_activation(r))
        centroid = np.asarray(response_weighted_centroid(r, pos), dtype=float)

        out.append(ClusterFeature(
            cluster_id=cid, center_xy=center, centroid_xy=centroid,
            normal_proxy=normal, shear_xy=shear, shear_mag=shear_mag,
            activation=wsum, n_members=len(members),
        ))
    return out
