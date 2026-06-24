"""Topology-blind scalar interpolation for HeatmapLayer.

Inverse-Distance Weighting (IDW) over the descriptor's PCA-plane positions. IDW
is the chosen Phase-2 interpolator because it is the only common option that
survives EVERY layout with no per-device branching:

  * a collinear LINE (RobotFinger_v1) - where Delaunay triangulation degenerates,
  * a sparse pair / quad (Mark2_tiny) - where an RBF solve is ill-conditioned,
  * a dense plane (Mark2_v1).

It needs no matrix solve, is deterministic, runs O(grid x sensors), and the
support mask keeps it honest: cells farther than a descriptor-derived radius
from any sensor stay NaN, so empty space is never painted (never fabricate gaps).
This module is pure numpy - no rendering backend, no descriptor import.
"""
from __future__ import annotations

import numpy as np


def _support_radius(xy: np.ndarray) -> float | None:
    """Median nearest-neighbor spacing of the sensor cloud (metres). None for a
    single point. This is the natural length scale of the layout, derived from
    positions - never a hardcoded constant."""
    n = len(xy)
    if n < 2:
        return None
    d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)
    nn = nn[np.isfinite(nn)]
    return float(np.median(nn)) if len(nn) else None


def idw_grid(
    xy: np.ndarray,
    values: np.ndarray,
    *,
    resolution: int = 80,
    power: float = 2.0,
    support_factor: float = 2.5,
) -> dict | None:
    """Interpolate `values` sampled at `xy` onto a regular grid by IDW.

    Non-finite samples (invalid / NaN sensors) are dropped before interpolation -
    they are never fabricated back. Returns a renderer-agnostic dict:
        {"grid": (R, R) float with NaN outside support, "extent": (x0,x1,y0,y1)}
    or None when there is no finite sample to interpolate from.
    """
    xy = np.asarray(xy, dtype=float)
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    xy, values = xy[finite], values[finite]
    if len(xy) == 0:
        return None

    mins, maxs = xy.min(axis=0), xy.max(axis=0)
    span = maxs - mins
    fallback = float(np.max(span)) or 1.0          # zero-area cloud (line/point)
    pad = np.where(span > 0, span * 0.10, fallback * 0.5)
    lo, hi = mins - pad, maxs + pad

    gx = np.linspace(lo[0], hi[0], resolution)
    gy = np.linspace(lo[1], hi[1], resolution)
    GX, GY = np.meshgrid(gx, gy)                     # (R, R)
    G = np.column_stack([GX.ravel(), GY.ravel()])    # (P, 2)

    d = np.linalg.norm(G[:, None, :] - xy[None, :, :], axis=2)   # (P, N)
    nn = d.min(axis=1)                                            # (P,)
    w = 1.0 / np.power(d + 1e-12, power)                          # exact hits -> huge w
    grid = (w * values[None, :]).sum(axis=1) / w.sum(axis=1)      # (P,)

    base = _support_radius(xy)
    if base is None:                                 # single sensor: fall back to pad
        base = fallback * 0.5
    grid = np.where(nn <= base * support_factor, grid, np.nan)

    return {
        "grid": grid.reshape(resolution, resolution),
        "extent": (float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])),
    }
