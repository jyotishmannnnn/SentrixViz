# SentrixViz Phase 2 — topology-aware tactile field visualization

> Builds **only** on the frozen 5-contract boundary (RenderModel, FieldSet,
> FrameSource, Layer, Renderer). No contract is changed. Phase 2 = new `Layer`
> impls + the minimum renderer primitive support + a `--view` CLI switch.

## 1. Where Phase 2 plugs in (no duplication)

| Existing seam | Phase 2 use |
|---|---|
| `RenderModel.project_2d()` (PCA plane) | reused verbatim as the layout for every layer — no new projection |
| `RenderModel.clusters` / `Node.cluster_id` | the ONLY source of cluster membership |
| `FieldSet.scalar/magnitude` | the per-sensor response scalar |
| `Layer.encode(model, fieldset) -> dict` | unchanged signature; new layers emit new primitive keys |
| `MatplotlibRenderer` | +3 optional primitive branches; existing branches untouched |
| primitives dict (`points`/`segments`) | +`field`, +`vectors`, +`markers` (additive keys) |

No new contract. `FieldLayer`/`TopologyLayer` are not modified.

## 2. New Layers (renderer-independent — emit physical scalars + geometry only)

All four read `(RenderModel, FieldSet)` and emit primitives; none import a backend.

- **HeatmapLayer** — `(model, fieldset)` → `field` raster. Per-node response
  scalar interpolated over the PCA plane by IDW; masked outside sensor support.
- **ClusterLayer** — per descriptor cluster: glyph at the cluster's geometric
  centre, **colour = normal_proxy**, **size = activation**, **arrow = shear**.
- **CentroidLayer** — per cluster: response-weighted contact **centroid**
  position, marker **sized by activation**.
- **ShearLayer** — per cluster: lateral-field **shear vector** as an arrow.

Cluster math is factored into one ephemeral helper `layers/derived.py`
(`cluster_features`) that mirrors the canonical formulas in SentrixDataEngine's
`derived.py` (normal_proxy / shear / centroid). It is **recomputed, never
imported and never persisted** (hard rules #3, #6). Single-frame simplification:
the canonical time-median baseline `B0` is taken as 0 in Phase 2 (a Layer sees
one frame); documented, and the shared pure-function reuse is the Phase-5 seam.

## 3. Heatmap interpolation — choice: **Inverse-Distance Weighting (IDW)**

| Option | Verdict for a topology-blind SDK |
|---|---|
| nearest-neighbor | blocky Voronoi; trivial but no gradient |
| **IDW** | **chosen** — any point set incl. a collinear line, no matrix solve, deterministic, O(N·grid), tunable power, masked outside support |
| RBF | needs an N×N solve; oscillates / ill-conditioned for sparse or collinear layouts; not robust |
| Delaunay triangulation | **degenerates for collinear points** → RobotFinger_v1 is a LINE → fails → breaks topology independence |
| graph interpolation | needs a connected dense graph; sparse clusters disconnect |

IDW is the only option that survives line (finger), plane (glove), and patch
with **zero per-device branching**, and is deterministic + fast enough for
playback. Support mask radius = `2.5 × median nearest-neighbor spacing` (derived
from positions, never hardcoded); cells beyond it stay NaN (no fabricated gaps).

## 4. Cluster visualization
Membership from `model.clusters[*].members` / `Node.cluster_id` ONLY. Visualizes
cluster activation (`normal_proxy`), shear magnitude/vector. Descriptors with no
clusters fall back to one-bucket-per-sensor (mirrors the canonical exporter).

## 5. Centroid visualization
Response-weighted centroid (`sum_k r_k·pos_k / sum_k r_k`) per cluster, weighting
the marker by total activation. Position comes from the descriptor (topology-driven).

## 6. Renderer changes (minimum)
Add three optional draw branches to `MatplotlibRenderer`: `field` (imshow +
colorbar), `vectors` (quiver), `markers` (scatter; size←weight, colour←values).
The renderer stays the sole owner of colour/size mapping. No redesign.

## 7. CLI additions
`sentrixviz render ... --view {raw|clusters|heatmap|centroid|shear}` (default
`raw` == existing behaviour, fully backward-compatible). View → layer stack:
raw→FieldLayer, heatmap→HeatmapLayer, clusters→ClusterLayer,
centroid→HeatmapLayer+CentroidLayer, shear→ShearLayer.

## 8. Testing
`tests/test_phase2_layers.py` + `tests/test_phase2_render.py`, descriptor-
parametrized over Mark2_v1 / Mark2_tiny / RobotFinger_v1:
- count independence — heatmap grid shape constant across 21/7/5 sensors.
- topology independence — IDW heatmap finite for the collinear RobotFinger line.
- descriptor independence — cluster-feature ids equal `desc.clusters` (or per-
  sensor fallback), proving membership is descriptor-driven, never name-parsed.
- every view renders valid PNG bytes for every descriptor.
- anti-hardcode: counts asserted against the descriptor, not literals.

## 9. Deliverables
- `layers/interpolate.py` — pure IDW grid.
- `layers/derived.py` — ephemeral `cluster_features`.
- `layers/heatmap.py` — `HeatmapLayer`.
- `layers/cluster.py` — `ClusterLayer`, `CentroidLayer`, `ShearLayer`.
- `render/mpl.py` — +field/vectors/markers branches.
- `cli.py` — `--view`.
- `layers/__init__.py` — exports + primitives schema doc.
- tests + this plan + ARCHITECTURE/CLAUDE phase-map updates.

Acceptance: descriptor → RenderModel → FieldSet → topology-aware Layer →
Renderer → PNG for all three descriptors with zero hardware special cases; the
parametrized suite (old 27 + new) is green.
