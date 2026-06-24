"""FieldLayer - Tier 1 raw values: each sensor node colored by a scalar drawn
from the frame (a raw channel like 'bz', or the vector magnitude 'mag').

Invalid sensors (valid flag False) are passed through with their value but
flagged so the renderer can mark them as "no data" - we never interpolate.
"""
from __future__ import annotations

import numpy as np

from ..core.fields import FieldSet
from ..core.model import RenderModel


class FieldLayer:
    def __init__(self, reduction: str = "mag", modality: str | None = None):
        self.reduction = reduction
        self.modality = modality
        self.name = f"field:{reduction}"

    def encode(self, model: RenderModel, fieldset: FieldSet | None) -> dict:
        if fieldset is None:
            raise ValueError("FieldLayer requires a fieldset")
        xy = model.project_2d()
        ids = [
            n.sensor_id for n in model.nodes
            if n.sensor_id in xy and (self.modality is None or n.modality == self.modality)
        ]
        coords = np.asarray([xy[i] for i in ids]) if ids else np.zeros((0, 2))
        values = np.asarray([fieldset.scalar(i, self.reduction) for i in ids], dtype=float)
        valid = np.asarray([fieldset.valid.get(i, True) for i in ids], dtype=bool)
        return {
            "title": f"{model.descriptor_version} - {self.name} "
                     f"@ t={fieldset.t_capture_us}us frame={fieldset.frame_index}",
            "points": {
                "key": f"points:{self.name}",
                "xy": coords,
                "ids": ids,
                "kind": "scalar",
                "values": values,
                "valid": valid,
            },
        }
