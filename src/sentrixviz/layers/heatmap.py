"""HeatmapLayer - Tier 2 continuous field: per-sensor response interpolated over
the descriptor's PCA plane by IDW (see `interpolate.py` for the choice rationale).

Emits a `field` raster primitive (renderer paints it) plus the sensor locations
as small `markers` so the support is visible. Invalid sensors are dropped from
the interpolation, never fabricated. Works for any sensor count and any layout -
line, plane, or patch - with no per-device branch.
"""
from __future__ import annotations

import numpy as np

from ..core.fields import FieldSet
from ..core.model import RenderModel
from .interpolate import idw_grid


class HeatmapLayer:
    def __init__(
        self,
        reduction: str = "mag",
        modality: str | None = None,
        resolution: int = 80,
        power: float = 2.0,
    ):
        self.reduction = reduction
        self.modality = modality
        self.resolution = resolution
        self.power = power
        self.name = f"heatmap:{reduction}"

    def encode(self, model: RenderModel, fieldset: FieldSet | None) -> dict:
        if fieldset is None:
            raise ValueError("HeatmapLayer requires a fieldset")
        xy = model.project_2d()
        ids = [
            n.sensor_id for n in model.nodes
            if n.sensor_id in xy and (self.modality is None or n.modality == self.modality)
        ]
        coords = np.asarray([xy[i] for i in ids]) if ids else np.zeros((0, 2))
        values = np.asarray([fieldset.scalar(i, self.reduction) for i in ids], dtype=float)
        valid = np.asarray([fieldset.valid.get(i, True) for i in ids], dtype=bool)
        values = np.where(valid, values, np.nan)

        out: dict = {
            "title": f"{model.descriptor_version} - {self.name} "
                     f"@ t={fieldset.t_capture_us}us frame={fieldset.frame_index}",
            "markers": {"key": f"markers:{self.name}", "xy": coords,
                        "style": "dot", "weight": None, "values": None},
        }
        field = idw_grid(coords, values, resolution=self.resolution, power=self.power)
        if field is not None:
            field["key"] = f"field:{self.name}"
            out["field"] = field
        return out
