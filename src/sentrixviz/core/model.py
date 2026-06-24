"""RenderModel - the canonical visualization scene, built ONLY from a Descriptor.

This is the L0 boundary of the SDK: descriptor in, renderable geometry out. It
carries every spatial fact the renderer needs (node positions, neighbor edges,
cluster groupings, frame, bounds) so that NO downstream code holds a geometry
constant. A new hardware revision is a new descriptor, never a code change.

What this module must NEVER do:
  - hardcode a sensor count, a finger name, or a fixed layout,
  - import a renderer, a UI framework, pyarrow, or matplotlib,
  - read a frame / value (that is FieldSet's job).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sentrix_contracts import Descriptor


@dataclass(frozen=True)
class Node:
    """One sensor as a renderable point. Position-bearing facts only."""
    sensor_id: str
    modality: str
    channels: tuple[str, ...]
    position_m: tuple[float, float, float] | None
    cluster_id: str | None


@dataclass(frozen=True)
class ClusterView:
    id: str
    members: tuple[str, ...]
    geometry: str | None
    rasterization: dict | None


@dataclass(frozen=True)
class RenderModel:
    """Static, hashable scene derived from one descriptor revision.

    Keyed by descriptor_hash so a viewer can cache projections / auto-ranges per
    hardware revision and invalidate them when the descriptor changes.
    """
    descriptor_version: str
    descriptor_hash: str | None
    device_class: str
    modalities: tuple[str, ...]
    nodes: tuple[Node, ...]                      # descriptor insertion order
    edges: tuple[tuple[str, str], ...]           # undirected neighbor pairs (sorted, unique)
    clusters: tuple[ClusterView, ...]
    regions: dict[str, tuple[str, ...]]
    frame: dict

    # ---- construction ---------------------------------------------------
    @classmethod
    def from_descriptor(cls, desc: Descriptor) -> "RenderModel":
        nodes = tuple(
            Node(
                sensor_id=s.sensor_id,
                modality=s.modality,
                channels=tuple(s.channels),
                position_m=tuple(s.position_m) if s.position_m is not None else None,
                cluster_id=s.cluster_id,
            )
            for s in desc.sensors.values()
        )
        # Edges come from desc.neighbors(), which already falls back to
        # radius-derived neighbors when no explicit graph is present. We never
        # compute geometry here; we only de-duplicate the descriptor's answer.
        seen: set[tuple[str, str]] = set()
        for sid in desc.sensors:
            for nb in desc.neighbors(sid):
                pair = tuple(sorted((sid, nb)))
                seen.add(pair)  # type: ignore[arg-type]
        edges = tuple(sorted(seen))
        clusters = tuple(
            ClusterView(id=c.id, members=tuple(c.members),
                        geometry=c.geometry, rasterization=c.rasterization)
            for c in desc.clusters.values()
        )
        return cls(
            descriptor_version=desc.descriptor_version,
            descriptor_hash=desc.descriptor_hash,
            device_class=desc.device_class,
            modalities=tuple(desc.modalities),
            nodes=nodes,
            edges=edges,
            clusters=clusters,
            regions=dict(desc.regions),
            frame=dict(desc.frame),
        )

    # ---- queries --------------------------------------------------------
    def node(self, sensor_id: str) -> Node:
        for n in self.nodes:
            if n.sensor_id == sensor_id:
                return n
        raise KeyError(sensor_id)

    def positioned_ids(self) -> list[str]:
        return [n.sensor_id for n in self.nodes if n.position_m is not None]

    def project_2d(self) -> dict[str, np.ndarray]:
        """Descriptor-driven 2D layout: principal plane of the sensor positions.

        Works for a line (finger), a plane (glove/palm), or a patch (skin) with
        no per-device special case. Sensors without a position are omitted. The
        result is centered; renderers add their own padding. This is the only
        place a 3D->2D decision is made, and it is data-driven (PCA), never a
        hardcoded axis pick.
        """
        ids = self.positioned_ids()
        if not ids:
            return {}
        pts = np.asarray([self.node(i).position_m for i in ids], dtype=float)
        centered = pts - pts.mean(axis=0, keepdims=True)
        # SVD principal axes; take the two with largest singular values.
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        basis = vh[:2].T  # (3, 2)
        xy = centered @ basis  # (N, 2)
        return {sid: xy[i] for i, sid in enumerate(ids)}
