"""TopologyLayer - static geometry: sensor nodes (colored by cluster) + neighbor
edges. Needs only a RenderModel; the fieldset is ignored. Drives `sentrixviz
topology` and proves descriptor-driven layout with no values present.
"""
from __future__ import annotations

import numpy as np

from ..core.fields import FieldSet
from ..core.model import RenderModel


class TopologyLayer:
    name = "topology"

    def encode(self, model: RenderModel, fieldset: FieldSet | None) -> dict:
        xy = model.project_2d()
        ids = [n.sensor_id for n in model.nodes if n.sensor_id in xy]
        coords = np.asarray([xy[i] for i in ids]) if ids else np.zeros((0, 2))

        # Cluster -> category index (None cluster -> its own bucket).
        cluster_order: list[str] = []
        cats = []
        for sid in ids:
            cid = model.node(sid).cluster_id or "(none)"
            if cid not in cluster_order:
                cluster_order.append(cid)
            cats.append(cluster_order.index(cid))

        index = {sid: k for k, sid in enumerate(ids)}
        seg0, seg1 = [], []
        for a, b in model.edges:
            if a in index and b in index:
                seg0.append(coords[index[a]])
                seg1.append(coords[index[b]])

        out: dict = {
            "title": f"{model.descriptor_version} ({model.device_class}) "
                     f"- {len(model.nodes)} sensors, {len(model.edges)} edges",
            "points": {
                "key": "points:topology",
                "xy": coords,
                "ids": ids,
                "kind": "category",
                "categories": np.asarray(cats, dtype=int),
                "category_labels": cluster_order,
                "valid": np.ones(len(ids), dtype=bool),
            },
        }
        if seg0:
            out["segments"] = {"xy0": np.asarray(seg0), "xy1": np.asarray(seg1)}
        return out
