"""NormalizationPolicy - the seam that decides what physical range maps to the
renderer's colour / size / vector-length / category channels.

WHY this exists
    A Layer emits PHYSICAL scalars; the Renderer maps them to appearance. Until
    now that mapping was implicit: matplotlib autoscaled every draw call to the
    one frame in front of it. Across a sequence the range moved every frame, so
    a heatmap flickered, centroid glyphs pulsed, shear arrows breathed, and
    cluster colours reshuffled. Normalization is the missing decision: "which
    range maps to the colormap endpoints", lifted out as an explicit policy.

WHERE it sits (no frozen contract is touched)
    - Layer  : still emits physical scalars. It only TAGS each primitive with a
               stable `key` so a policy can address it. Additive; backend-blind.
    - Renderer: still the sole owner of appearance. It now ASKS a policy for the
               range instead of letting the backend autoscale. The policy lives
               on the renderer like `cmap` does (constructor), not in render()'s
               frozen signature.
    - Timeline/orchestration: the only thing that sees every frame, so it SCANS
               them (`scan_ranges`) to build a global policy. It computes
               physical statistics, never colours.

Strategy interface, four channels (one per flicker source):
    color_range  -> (vmin, vmax)  for field / scalar points / marker values
    size_range   -> (lo, hi)      for marker weights -> glyph area
    vector_scale -> float         for quiver arrow length (data units per xy)
    categories   -> [label,...]   stable label order -> stable category colour

A neutral return (None / (None, None)) means "autoscale this call" - which is
exactly the legacy behaviour, so PerFrameNormalizer is byte-identical to today.

Shipped now: PerFrameNormalizer, GlobalNormalizer, scan_ranges, RollingNormalizer.
Deferred (this is the seam they will plug into):
PercentileNormalizer (robust), DescriptorNormalizer (physical units).
"""
from __future__ import annotations

from typing import Iterable, Protocol, Sequence, runtime_checkable

import numpy as np

from .core.fields import FieldSet
from .core.model import RenderModel
from .core.protocols import Layer

# A range whose hi <= lo is degenerate (constant field). Reporting it as "no
# range" makes the renderer autoscale that one call - which for a constant field
# is identical anyway - and avoids a vmin==vmax matplotlib degeneracy.
_NONE_RANGE: tuple[float | None, float | None] = (None, None)


@runtime_checkable
class NormalizationPolicy(Protocol):
    """Resolves the appearance range for a primitive, addressed by its `key`.

    `values`/`weights`/`uv` are the CURRENT frame's data: a per-frame policy uses
    them; a global policy ignores them and returns its precomputed answer. Either
    way a neutral return means "let the backend autoscale this call"."""

    def color_range(self, key: str | None, values: np.ndarray) -> tuple[float | None, float | None]: ...

    def size_range(self, key: str | None, weights: np.ndarray) -> tuple[float | None, float | None]: ...

    def vector_scale(self, key: str | None, uv: np.ndarray) -> float | None: ...

    def categories(self, key: str | None) -> list[str] | None: ...


class PerFrameNormalizer:
    """The legacy behaviour, made explicit: every call autoscales to its own
    frame. Returns neutral for every channel, so the renderer falls back to the
    backend's per-call autoscale - byte-identical to pre-seam output. This is the
    default single-frame policy."""

    def color_range(self, key: str | None, values: np.ndarray) -> tuple[float | None, float | None]:
        return _NONE_RANGE

    def size_range(self, key: str | None, weights: np.ndarray) -> tuple[float | None, float | None]:
        return _NONE_RANGE

    def vector_scale(self, key: str | None, uv: np.ndarray) -> float | None:
        return None

    def categories(self, key: str | None) -> list[str] | None:
        return None


class GlobalNormalizer:
    """A fixed policy: one range per key, the same for every frame. Built by
    `scan_ranges` over a whole sequence, so playback is stable end to end. Holds
    only physical statistics + label orders - never a colour."""

    def __init__(
        self,
        color_ranges: dict[str, tuple[float, float]] | None = None,
        size_ranges: dict[str, tuple[float, float]] | None = None,
        vector_scales: dict[str, float] | None = None,
        category_maps: dict[str, list[str]] | None = None,
    ):
        self._color = dict(color_ranges or {})
        self._size = dict(size_ranges or {})
        self._vec = dict(vector_scales or {})
        self._cats = dict(category_maps or {})

    def color_range(self, key: str | None, values: np.ndarray) -> tuple[float | None, float | None]:
        return self._color.get(key, _NONE_RANGE)  # type: ignore[arg-type]

    def size_range(self, key: str | None, weights: np.ndarray) -> tuple[float | None, float | None]:
        return self._size.get(key, _NONE_RANGE)  # type: ignore[arg-type]

    def vector_scale(self, key: str | None, uv: np.ndarray) -> float | None:
        return self._vec.get(key)  # type: ignore[arg-type]

    def categories(self, key: str | None) -> list[str] | None:
        return self._cats.get(key)  # type: ignore[arg-type]


# ---- live (rolling) -----------------------------------------------------------

class RollingNormalizer:
    """Live policy: bounded-memory auto-gain over a push stream.

    `GlobalNormalizer` needs a whole-stream scan, which a live source cannot
    provide. `RollingNormalizer` instead adapts as frames are queried, holding
    ONE `[lo, hi]` per key - so memory is O(keys), independent of stream length
    (the bounded-memory requirement). It plugs into the same seam and mutates
    its own state on each `color_range`/`size_range`/`vector_scale` call.

    Auto-gain keeps visuals stable AND adaptive:
      - EXPAND immediately when a frame exceeds the current range, so a sudden
        contact peak is never clipped;
      - RELAX slowly (EMA, rate `alpha`) back toward the frame range, so the
        range never collapses on a quiet frame and the colormap does not flicker.
    A smaller `alpha` is steadier (slower to forget an old extreme); a larger
    `alpha` tracks the live signal more tightly.

    Vector scale needs the (frame-constant) anchor-cloud span, which is NOT in
    the per-call `uv`. The orchestrator primes it once via `prime_vector_span`
    (positions never move); until primed, vector scale falls back to None
    (the renderer autoscales that call - legacy behaviour, no crash).
    """

    def __init__(self, alpha: float = 0.1):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self._alpha = float(alpha)
        self._color: dict[str, list[float]] = {}   # key -> [lo, hi]
        self._size: dict[str, list[float]] = {}     # key -> [lo, hi]
        self._vec_mag: dict[str, float] = {}        # key -> EMA of max |uv|
        self._vec_span: dict[str, float] = {}       # key -> primed anchor span
        self._cats: dict[str, list[str]] = {}       # key -> ordered label union

    # ---- adaptation core ------------------------------------------------
    def _adapt(self, table: dict[str, list[float]], key: str,
               lo: float, hi: float) -> tuple[float | None, float | None]:
        cur = table.get(key)
        if cur is None:
            table[key] = [lo, hi]
        else:
            a = self._alpha
            # expand instantly (fast attack), relax via EMA (slow release)
            cur[0] = min(lo, cur[0] + a * (lo - cur[0]))
            cur[1] = max(hi, cur[1] + a * (hi - cur[1]))
        lo_o, hi_o = table[key]
        return (lo_o, hi_o) if hi_o > lo_o else _NONE_RANGE

    # ---- NormalizationPolicy --------------------------------------------
    def color_range(self, key: str | None, values: np.ndarray) -> tuple[float | None, float | None]:
        if key is None:
            return _NONE_RANGE
        f = _finite(values)
        if f.size:
            return self._adapt(self._color, key, float(f.min()), float(f.max()))
        cur = self._color.get(key)
        return (cur[0], cur[1]) if cur and cur[1] > cur[0] else _NONE_RANGE

    def size_range(self, key: str | None, weights: np.ndarray) -> tuple[float | None, float | None]:
        if key is None:
            return _NONE_RANGE
        f = _finite(weights)
        if f.size:
            return self._adapt(self._size, key, float(f.min()), float(f.max()))
        cur = self._size.get(key)
        return (cur[0], cur[1]) if cur and cur[1] > cur[0] else _NONE_RANGE

    def vector_scale(self, key: str | None, uv: np.ndarray) -> float | None:
        if key is None:
            return None
        mags = np.linalg.norm(np.asarray(uv, dtype=float).reshape(-1, 2), axis=1)
        mags = mags[np.isfinite(mags)]
        if mags.size:
            m = float(mags.max())
            cur = self._vec_mag.get(key)
            # same auto-gain on magnitude: grow fast, shrink slowly
            self._vec_mag[key] = m if cur is None else max(m, cur + self._alpha * (m - cur))
        m = self._vec_mag.get(key, 0.0)
        span = self._vec_span.get(key)
        if m > 0 and span:
            return m / (0.25 * span)   # biggest arrow ~25% of the anchor cloud
        return None

    def categories(self, key: str | None) -> list[str] | None:
        if key is None:
            return None
        return self._cats.get(key)

    # ---- priming (static, frame-independent facts) ----------------------
    def prime_vector_span(self, key: str, span: float) -> None:
        """Seed the anchor-cloud span for a vector key (computed once from the
        model; positions never move). Lets vector_scale produce a stable arrow
        length without seeing every frame."""
        if span and span > 0:
            self._vec_span[key] = float(span)

    def observe_categories(self, key: str, labels: list[str]) -> None:
        """Accumulate the label order for a category key as new clusters appear,
        so a cluster keeps its colour once seen (bounded by #distinct labels)."""
        bucket = self._cats.setdefault(key, [])
        for lab in labels:
            if lab not in bucket:
                bucket.append(lab)


# ---- scanning -----------------------------------------------------------------

def _finite(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float).ravel()
    return a[np.isfinite(a)]


class _Accum:
    """Running min/max over finite samples. Stays empty (-> no range) until a
    finite sample arrives, so an all-NaN key never fabricates a range."""

    def __init__(self) -> None:
        self.lo = np.inf
        self.hi = -np.inf

    def add(self, vals: np.ndarray) -> None:
        f = _finite(vals)
        if f.size:
            self.lo = min(self.lo, float(f.min()))
            self.hi = max(self.hi, float(f.max()))

    def range(self) -> tuple[float, float] | None:
        # hi > lo required: a flat field is reported as no-range (autoscale, which
        # is identical for a constant) to dodge a vmin==vmax degeneracy.
        if self.hi > self.lo:
            return (self.lo, self.hi)
        return None


def scan_ranges(
    model: RenderModel,
    layers: Sequence[Layer],
    fieldsets: Iterable[FieldSet],
) -> GlobalNormalizer:
    """One pass over `fieldsets`: encode every layer per frame and accumulate the
    physical range of each tagged primitive into a frame-stable GlobalNormalizer.

    Topology-blind and count-blind: it reads whatever primitives the layers emit
    and keys them by the layers' own `key` tags. Untagged primitives are skipped
    (their key is None -> the renderer autoscales them, i.e. no regression)."""
    color: dict[str, _Accum] = {}
    size: dict[str, _Accum] = {}
    vec_mag: dict[str, float] = {}      # global max |uv| per key
    vec_span: dict[str, float] = {}     # anchor-cloud span per key (xy units)
    cats: dict[str, list[str]] = {}     # ordered union of labels per key

    def acc(table: dict[str, _Accum], key: str) -> _Accum:
        return table.setdefault(key, _Accum())

    for fs in fieldsets:
        for layer in layers:
            prims = layer.encode(model, fs)

            field = prims.get("field")
            if field is not None and field.get("key") is not None:
                acc(color, field["key"]).add(field["grid"])

            pts = prims.get("points")
            if pts is not None and pts.get("key") is not None:
                if pts.get("kind") == "category":
                    bucket = cats.setdefault(pts["key"], [])
                    for lab in pts.get("category_labels", []):
                        if lab not in bucket:
                            bucket.append(lab)
                else:
                    valid = pts.get("valid")
                    vals = np.asarray(pts.get("values"), dtype=float)
                    if valid is not None:
                        vals = np.where(np.asarray(valid, dtype=bool), vals, np.nan)
                    acc(color, pts["key"]).add(vals)

            marks = prims.get("markers")
            if marks is not None and marks.get("key") is not None:
                if marks.get("values") is not None:
                    acc(color, marks["key"]).add(np.asarray(marks["values"], dtype=float))
                if marks.get("weight") is not None:
                    acc(size, marks["key"]).add(np.asarray(marks["weight"], dtype=float))

            vecs = prims.get("vectors")
            if vecs is not None and vecs.get("key") is not None:
                k = vecs["key"]
                uv = np.asarray(vecs.get("uv"), dtype=float)
                if uv.size:
                    mags = np.linalg.norm(uv, axis=1)
                    mags = mags[np.isfinite(mags)]
                    if mags.size:
                        vec_mag[k] = max(vec_mag.get(k, 0.0), float(mags.max()))
                xy = np.asarray(vecs.get("xy"), dtype=float)
                if xy.size:
                    span = float(np.nanmax(xy.max(axis=0) - xy.min(axis=0)))
                    vec_span[k] = max(vec_span.get(k, 0.0), span)

    color_ranges = {k: r for k, a in color.items() if (r := a.range()) is not None}
    size_ranges = {k: r for k, a in size.items() if (r := a.range()) is not None}

    # quiver with scale_units="xy": arrow length (xy units) = |uv| / scale. Pick
    # scale so the GLOBAL biggest arrow spans ~25% of the (frame-constant) anchor
    # cloud - same scale every frame, so arrows stop breathing.
    vector_scales: dict[str, float] = {}
    for k, m in vec_mag.items():
        span = vec_span.get(k, 0.0) or 1.0
        if m > 0:
            vector_scales[k] = m / (0.25 * span)

    return GlobalNormalizer(
        color_ranges=color_ranges,
        size_ranges=size_ranges,
        vector_scales=vector_scales,
        category_maps=cats,
    )
