"""MatplotlibRenderer - the minimal viable renderer (headless PNG).

Chosen as the Phase-1 renderer because it is the smallest thing that exercises
the WHOLE SDK boundary (descriptor -> model -> projection -> layer primitives ->
pixels) while staying CI-friendly: no GPU, no browser, no event loop, and PNG
bytes are byte-assertable in tests. It is deliberately NOT the long-term
frontend; swapping in WebGL means replacing this file and nothing upstream.

The renderer is the ONLY component that maps scalars/categories to colors.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from ..core.fields import FieldSet
from ..core.model import RenderModel
from ..core.protocols import Layer
from ..normalize import NormalizationPolicy, PerFrameNormalizer


class MatplotlibRenderer:
    def __init__(
        self,
        dpi: int = 110,
        point_size: float = 120.0,
        cmap: str = "viridis",
        normalizer: NormalizationPolicy | None = None,
    ):
        self.dpi = dpi
        self.point_size = point_size
        self.cmap = cmap
        # The renderer stays the sole owner of appearance; the policy only tells
        # it which physical range maps to the colour/size/length endpoints. None
        # means per-frame autoscale - byte-identical to the pre-seam renderer.
        self.normalizer: NormalizationPolicy = normalizer or PerFrameNormalizer()

    def set_normalizer(self, normalizer: NormalizationPolicy) -> None:
        """Swap the policy in place (used by sequence rendering to apply a
        scanned GlobalNormalizer for the duration of an export)."""
        self.normalizer = normalizer

    def render(
        self,
        model: RenderModel,
        layers: Sequence[Layer],
        fieldset: FieldSet | None = None,
        out: Path | None = None,
    ) -> bytes | Path:
        import io

        import matplotlib
        matplotlib.use("Agg")  # headless; no display required
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 6), dpi=self.dpi)
        ax.set_aspect("equal")
        title = model.descriptor_version

        for layer in layers:
            prims = layer.encode(model, fieldset)
            title = prims.get("title", title)
            field = prims.get("field")
            if field is not None:
                self._draw_field(ax, field)
            segs = prims.get("segments")
            if segs is not None and len(segs["xy0"]):
                for p0, p1 in zip(segs["xy0"], segs["xy1"]):
                    ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
                            color="0.7", lw=0.8, zorder=1)
            pts = prims.get("points")
            if pts is not None and len(pts["xy"]):
                self._draw_points(ax, pts)
            vecs = prims.get("vectors")
            if vecs is not None and len(vecs["xy"]):
                self._draw_vectors(ax, vecs)
            marks = prims.get("markers")
            if marks is not None and len(marks["xy"]):
                self._draw_markers(ax, marks)

        self._guard_degenerate_axes(ax)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()

        if out is not None:
            out = Path(out)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, format="png")
            plt.close(fig)
            return out
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        return buf.getvalue()

    def _draw_points(self, ax, pts: dict) -> None:
        import matplotlib.pyplot as plt

        xy = pts["xy"]
        key = pts.get("key")
        valid = pts.get("valid", np.ones(len(xy), dtype=bool))
        if pts["kind"] == "scalar":
            vals = pts["values"].copy()
            finite = np.isfinite(vals) & valid
            vmin, vmax = self.normalizer.color_range(key, vals[finite])
            sc = ax.scatter(
                xy[finite, 0], xy[finite, 1],
                c=vals[finite], cmap=self.cmap, s=self.point_size,
                edgecolors="k", linewidths=0.4, zorder=2, vmin=vmin, vmax=vmax,
            )
            ax.figure.colorbar(sc, ax=ax, shrink=0.7)
            # invalid / non-finite sensors: drawn hollow, never interpolated
            bad = ~finite
            if bad.any():
                ax.scatter(xy[bad, 0], xy[bad, 1], facecolors="none",
                           edgecolors="0.4", s=self.point_size, zorder=2)
        else:  # category
            cats = np.asarray(pts["categories"], dtype=int)
            labels = list(pts.get("category_labels", []))
            # A global category map pins every label to a fixed colour index, so
            # a frame showing only a subset of clusters keeps each cluster's
            # colour. Without it, fall back to the frame-local indices (legacy).
            gmap = self.normalizer.categories(key)
            cvmin = cvmax = None
            if gmap and labels:
                idx = {lab: gmap.index(lab) for lab in labels if lab in gmap}
                cats = np.asarray([idx.get(labels[c], c) for c in cats], dtype=int)
                labels = gmap
                cvmin, cvmax = 0, max(len(gmap) - 1, 1)
            sc = ax.scatter(xy[:, 0], xy[:, 1], c=cats, cmap="tab10",
                            s=self.point_size, edgecolors="k", linewidths=0.4,
                            zorder=2, vmin=cvmin, vmax=cvmax)
            if labels:
                handles = [
                    plt.Line2D([], [], marker="o", linestyle="",
                               markerfacecolor=sc.cmap(sc.norm(i)),
                               markeredgecolor="k", label=lab)
                    for i, lab in enumerate(labels)
                ]
                ax.legend(handles=handles, fontsize=7, loc="best", framealpha=0.6)

    @staticmethod
    def _guard_degenerate_axes(ax) -> None:
        """A single point or a collinear cloud (e.g. one cluster of a finger
        line) collapses an equal-aspect axes to zero width/height, which makes
        the data transform singular (quiver crashes). Pad such axes minimally."""
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        w, h = x1 - x0, y1 - y0
        eps = max(w, h, 1e-3) * 0.5
        if w <= 1e-9:
            cx = 0.5 * (x0 + x1)
            ax.set_xlim(cx - eps, cx + eps)
        if h <= 1e-9:
            cy = 0.5 * (y0 + y1)
            ax.set_ylim(cy - eps, cy + eps)

    # ---- Phase 2 primitives (renderer owns color/size mapping) -----------
    def _draw_field(self, ax, field: dict) -> None:
        """Interpolated raster. NaN cells (outside sensor support) stay blank."""
        grid = np.ma.masked_invalid(np.asarray(field["grid"], dtype=float))
        vmin, vmax = self.normalizer.color_range(field.get("key"), grid.compressed())
        im = ax.imshow(grid, extent=field["extent"], origin="lower",
                       cmap=self.cmap, alpha=0.9, aspect="equal",
                       interpolation="nearest", zorder=0, vmin=vmin, vmax=vmax)
        ax.figure.colorbar(im, ax=ax, shrink=0.7)

    def _draw_vectors(self, ax, vecs: dict) -> None:
        xy = np.asarray(vecs["xy"], dtype=float)
        uv = np.asarray(vecs["uv"], dtype=float)
        ok = np.isfinite(uv).all(axis=1) & (np.linalg.norm(uv, axis=1) > 0)
        if ok.any():
            scale = self.normalizer.vector_scale(vecs.get("key"), uv[ok])
            kw = dict(color="crimson", zorder=4, width=0.006)
            # A fixed scale (data units per xy) stops arrows breathing frame to
            # frame; None keeps matplotlib's per-frame autoscale (legacy).
            if scale is not None:
                kw.update(angles="xy", scale_units="xy", scale=scale)
            ax.quiver(xy[ok, 0], xy[ok, 1], uv[ok, 0], uv[ok, 1], **kw)

    def _draw_markers(self, ax, m: dict) -> None:
        xy = np.asarray(m["xy"], dtype=float)
        key = m.get("key")
        marker = "*" if m.get("style") == "star" else "o"
        weight = m.get("weight")
        w_arr = np.asarray(weight, dtype=float) if weight is not None else np.empty(0)
        size_lo, size_hi = self.normalizer.size_range(key, w_arr)
        size = self._weights_to_sizes(weight, base=self.point_size,
                                      lo=size_lo, hi=size_hi)
        values = m.get("values")
        if values is not None:
            vals = np.asarray(values, dtype=float)
            finite = np.isfinite(vals)
            if finite.any():
                s = size[finite] if np.ndim(size) else size
                vmin, vmax = self.normalizer.color_range(key, vals[finite])
                sc = ax.scatter(xy[finite, 0], xy[finite, 1], c=vals[finite],
                                cmap=self.cmap, s=s, marker=marker,
                                edgecolors="k", linewidths=0.5, zorder=3,
                                vmin=vmin, vmax=vmax)
                ax.figure.colorbar(sc, ax=ax, shrink=0.7)
                return
        ax.scatter(xy[:, 0], xy[:, 1], s=size, marker=marker, c="0.15",
                   edgecolors="k", linewidths=0.5, zorder=3)

    @staticmethod
    def _weights_to_sizes(weight, base: float, lo=None, hi=None):
        """Map a layer's physical weights to marker areas. None -> fixed size.
        `lo`/`hi` pin the range across frames (stable sizing); when None the
        range is taken from this frame's own weights (legacy per-frame)."""
        if weight is None:
            return base
        w = np.asarray(weight, dtype=float)
        finite = np.isfinite(w)
        if not finite.any():
            return base
        if lo is None or hi is None:
            lo, hi = float(np.min(w[finite])), float(np.max(w[finite]))
        rng = (hi - lo) or 1.0
        norm = np.nan_to_num((w - lo) / rng)         # NaN weights -> smallest
        return base * (0.6 + 1.8 * norm)
