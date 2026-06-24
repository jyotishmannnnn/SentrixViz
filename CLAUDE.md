# SentrixViz — Repo Memory

> Topology-driven visualization SDK for the Sentrix stack. Read the root
> `../CLAUDE.md` first for ecosystem-wide rules. This file is the repo-specific
> contract. Current maturity: **Phase 0 + Phase 1 only** (parquet playback +
> headless PNG). Later phases are noted, not built.

## Purpose

Turn any Sentrix tactile artifact into a picture, driven entirely by the
`TopologyDescriptor`. SentrixViz is a *reusable SDK capability*, not a glove
viewer: the same code renders Mark2_v1, Mark2_tiny, RobotFinger_v1, and future
robot/humanoid skins with **no code change** — only a new descriptor.

## Position in the ecosystem

```
Descriptor (sentrix_contracts) ─┐
raw.parquet (Sim | Capture) ─────┼──► SentrixViz ──► PNG (Phase 1) / live, web (later)
Sync outputs / derived exports ──┘   reads artifacts + the contract, never code
```

SentrixViz sits **beside** the pipeline, exactly as SentrixSync sits beside the
producers: it depends on *output artifacts* + `sentrix_contracts`, never on the
code of Sim / Capture / Sync / DataEngine.

## Hard rules (non-negotiable — inherited from root principle 8)

1. **Zero geometry constants.** Every spatial fact (position, neighbor, cluster,
   raster grid, frame) comes from a `Descriptor`, addressed by `descriptor_hash`.
   If a viewer needs a number not in the descriptor, that number belongs in the
   descriptor (or its calibration bundle), not here.
2. **No sensor-count / finger / device-class branching.** Counts come from
   `desc.ids()` / `len(desc.sensors)`. No `21`, no `"thumb"`, no `if glove`.
3. **No downstream/upstream code imports.** Never `import sentrixsim /
   sentrixcapture / sentrixsync / sentrixdataengine`. Consume their artifacts.
4. **Reuse the at-rest convention.** Columns are resolved with
   `sentrix_contracts.parse_column` / `column_for`. Do **not** invent a second
   visualization schema or re-key columns.
5. **Never fabricate gaps.** Invalid / NaN frames render as explicit "no data"
   (hollow markers), never interpolated. Mirrors root principle 5.
6. **Display signals are not canonical.** Any derived/reduced value computed for
   display is ephemeral; SentrixViz never writes a dataset or mutates an artifact.
7. **Live vs playback is a `FrameSource` distinction, not a renderer one.** Both
   emit `(timestamp, FieldSet)`; the renderer never knows the origin.

## The five contracts (SDK boundary)

| Contract | Kind | Lives in | Must never know |
|---|---|---|---|
| `RenderModel` | frozen dataclass | `core/model.py` | values, pixels, I/O |
| `FieldSet` | frozen dataclass | `core/fields.py` | positions, topology, drawing |
| `FrameSource` | Protocol | `core/protocols.py` (impl `sources/`) | positions, drawing |
| `Layer` | Protocol | `core/protocols.py` (impl `layers/`) | colors, backend |
| `Renderer` | Protocol | `core/protocols.py` (impl `render/`) | how frames were produced |

Boundary law: **everything up to "what scalar/category + where" is pure core; the
act of putting pixels on a surface is the Renderer.** Layers emit physical scalars
or category indices + geometry — never colors. The Renderer owns colormaps.

## Layout

```
src/sentrixviz/
  core/      pure SDK: model, fields, protocols   (deps: sentrix_contracts, numpy)
  sources/   FrameSource adapters: parquet (Phase 1)
  layers/    pure encoders: topology, field (Tier 1)
  render/    renderers: mpl (headless PNG, minimal viable)
  cli.py     inspect | topology | render
tests/       descriptor-parametrized; self-contained synth raw.parquet
```

`core/` must stay importable with only `sentrix_contracts` + `numpy`. pyarrow and
matplotlib are confined to `sources/` and `render/` respectively.

## Dev setup

`sentrix_contracts` is vendored in SentrixCapture for v0.1.0. Install it editable,
then this package:

```
pip install -e ../SentrixCapture/contracts/python
pip install -e .[test]
pytest
```

(The shared `SentrixSim/.venv` already carries numpy/pyarrow + the contracts;
add matplotlib + pytest there.)

## Tests prove (and must keep proving)

- **count independence** — Mark2_v1 (21+3) vs Mark2_tiny (7) vs RobotFinger_v1 (5).
- **descriptor independence** — a bundled descriptor, a local glove descriptor, a
  synthetic non-glove descriptor, all render via one code path.
- **topology independence** — line (finger) / plane (glove) layouts from the same
  PCA projection; edges from explicit graph or radius, untouched by viz.

A regression that hardcodes a count, a name, or a layout will break the
parametrized suite by construction.

## Future phases (NOTED — do not build without a new task)

- **P2 — BUILT.** Topology-aware layers: `HeatmapLayer` (IDW over the PCA plane,
  masked outside descriptor-derived support), `ClusterLayer` / `CentroidLayer` /
  `ShearLayer` (per-descriptor-cluster normal_proxy / shear / weighted centroid).
  Cluster membership comes only from the descriptor; the per-cluster math is an
  ephemeral copy of DataEngine's `derived.py` formulas (B0=0 single-frame),
  recomputed not imported. Renderer gained `field`/`vectors`/`markers`
  primitives; CLI gained `render --view {raw|heatmap|clusters|centroid|shear}`
  (raw == Phase-1 default). See `PHASE2_PLAN.md`.
- **P3 — BUILT.** Live (push) visualization. One new abstraction: `FrameFeed`
  (transport seam in `sources/live.py`) yielding `RawFrame`s; the transport
  (USB / socket / Capture bus / Sim) lives behind it and is never imported.
  `LiveFrameSource` is the iterable, descriptor-paired sibling of
  `ParquetFrameSource` (emits `FieldSet`s; NOT indexable — live has no length).
  Built-in feeds (no hardware): `ReplayFeed` (parquet→live, the parity oracle),
  `SyntheticFeed` (descriptor-driven), `QueueFeed` (latest-wins, never
  backpressures the recorder). `RollingNormalizer` adds bounded-memory adaptive
  range mapping in the existing `NormalizationPolicy` seam (expand-fast /
  relax-slow auto-gain). `LiveSession` (`live/`) is the unbounded latest-wins
  sibling of `render_sequence`, reusing the exact render path. CLI gained
  `live` (replay a parquet or `--synth`; `-o` monitor / `--out-dir` + manifest).
  No frozen contract touched. See `PHASE3_PLAN.md`.
- **P4** Sync-grid synchronized playback + confidence overlay + dropout rendering.
- **P5** Tier-3 derived features (shear/centroid/activation) reusing the shared
  derived math — one pure function, batch and live parity.
- **P6** WebGL frontend implementing `Renderer`; video-overlay + Tier-4 ML seams.
