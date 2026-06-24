# SentrixViz

Topology-driven visualization SDK for the Sentrix stack. One code path renders
**any** hardware revision — glove, finger module, future robot/humanoid skin —
driven entirely by its `TopologyDescriptor`. Descriptor in, picture out; zero
geometry constants, no sensor-count branching.

```
raw.parquet (SentrixSim | SentrixCapture)  ─┐
silver.parquet (SentrixSync, --silver)      ─┤→ SentrixViz → PNG · PNG sequence · MP4/GIF · live monitor
live FrameFeed (USB / socket / synth)       ─┘
                       ▲
                sentrix-contracts   (topology descriptor + at-rest column convention)
```

---

## Purpose

A reusable SDK capability, not a glove-specific viewer. SentrixViz consumes
Sentrix tactile artifacts plus a topology descriptor and emits rendered imagery:
static geometry, single frames, frame sequences, video, and a live monitor. It
sits *beside* the pipeline (like SentrixSync), never inside it — it imports no
producer or consumer code, only artifacts and `sentrix_contracts`.

## Responsibilities

- Consume descriptor + `raw.parquet` / `silver.parquet` artifacts (never producer code).
- Render topology geometry, frames, sequences, video, and live streams.
- Implement the five-contract SDK boundary (below) so new devices, sources,
  layers, and renderers plug in without touching the core.
- Enforce the hard rules: **zero geometry constants**, **no sensor-count
  branching**, **no producer imports**, reuse the at-rest column convention,
  never fabricate gaps, never persist display signals as canonical data, and
  keep live ≠ playback distinct in code.

## Public interfaces

**The five contracts** (`sentrixviz.core`):

| Contract | Role |
|---|---|
| `RenderModel` | Static scene from a descriptor — nodes, edges, clusters, regions, frame bounds. |
| `FieldSet` | One frame's sensor values bound to sensor ids (`scalar()`, `magnitude()`, `channel()`). |
| `FrameSource` | Descriptor-paired, timestamped stream of frames. |
| `Layer` | `(model, fieldset)` → renderer-agnostic visual primitives. |
| `Renderer` | Primitives → pixels. |

**Implementations shipped:** `ParquetFrameSource`, `SyncFrameSource`,
`LiveFrameSource` (sources); `TopologyLayer`, `FieldLayer`, `HeatmapLayer`,
`ClusterLayer`, `CentroidLayer`, `ShearLayer` (layers); `MatplotlibRenderer`
(headless PNG); `PerFrameNormalizer`, `GlobalNormalizer`, `RollingNormalizer`
(normalization policies); `FrameFeed`, `ReplayFeed`, `SyntheticFeed`, `QueueFeed`
(live transports). Helpers: `bind_row`, `scan_ranges`, `resolve_descriptor`,
`render_sequence`, `render_video`.

**CLI** (`sentrixviz`):

| Command | Does |
|---|---|
| `inspect RAW.parquet` | Summarize artifact: sensor counts, frames, duration, metadata. |
| `topology --descriptor D` | Static geometry PNG (nodes + edges by cluster). |
| `render RAW.parquet` | One-frame PNG; `--view {raw\|heatmap\|clusters\|centroid\|shear}`. |
| `render-sequence RAW.parquet` | Ordered PNG sequence + `manifest.json`. |
| `render-video` | MP4 (ffmpeg) or GIF (Pillow) from a sequence. |
| `live RAW.parquet \| --synth` | Live (push) visualization replay or synthetic feed. |

## Inputs

- **Artifacts:** `raw.parquet` (from SentrixSim or SentrixCapture) carrying
  self-describing schema metadata (`sentrix_descriptor_version`,
  `sentrix_descriptor_hash`, `ts_column`).
- **Descriptors:** `TopologyDescriptor` JSON — bundled in `sentrix_contracts`
  (e.g. `Mark2_v1`) or supplied via `--descriptor PATH`.
- **Sync output:** `silver.parquet` via `--silver` (read through `SyncFrameSource`).
- **Live transports:** any `FrameFeed` implementation (USB reader, socket,
  Capture bus, Sim stream) — pluggable, never imported.

## Outputs

- **Static:** PNG (topology geometry, single frames).
- **Sequences:** ordered PNG frames (`000000.png`, …) + `manifest.json` (frame
  index, descriptor hash, layer names).
- **Video:** MP4 (ffmpeg) or GIF (Pillow).
- **Live monitor:** latest-wins PNG (`-o latest.png`) and/or a numbered frame
  directory + manifest.
- Display values (scalars, category indices, geometry) are **ephemeral** — never
  persisted as canonical data, never mutate artifacts.

## Dependencies

- **`sentrix-contracts>=0.1.0`** — descriptor types, bundled versions,
  `parse_column` / `column_for` / `dyn_columns`.
- `numpy>=1.24`, `pyarrow>=14` (artifact I/O), `matplotlib>=3.7` (headless PNG;
  used only by `MatplotlibRenderer`, not by `core/`).
- Dev/test: `pytest>=7` (`dev` extra; `test` is a back-compat alias).

No imports from SentrixSim, SentrixCapture, SentrixSync, or SentrixDataEngine.

## Quick start

```bash
# sentrix-contracts is a standalone package; install editable if not on an index
pip install -e ../SentrixContracts
pip install -e ".[dev]"

sentrixviz inspect raw.parquet                                  # summarize an artifact
sentrixviz topology --descriptor Mark2_v1 -o topo.png           # static geometry
sentrixviz render raw.parquet --view heatmap -o frame.png       # one frame
sentrixviz render-sequence raw.parquet --out seq/               # PNG sequence + manifest
sentrixviz render-video --seq seq/ -o out.mp4                   # encode sequence
sentrixviz live --synth                                         # live synthetic feed
```

**Adding a new hardware revision:** drop a descriptor and render unchanged — no
code edit. The same path renders Mark2_v1 (21+3), Mark2_tiny, and RobotFinger_v1.

## Testing

```bash
pip install -e ".[dev]"
pytest
```

11 test modules, **descriptor-parametrized** across `Mark2_v1`, `Mark2_tiny`, and
`RobotFinger_v1`. The suite asserts count independence, descriptor independence
(bundled / local / synthetic), and topology independence (line/plane layouts,
edges from explicit graph or radius). Any regression that hardcodes a sensor
count, name, or layout fails by construction. A synthetic raw-parquet fixture
(`conftest.py`) mirrors the producer at-rest contract, so the SDK is tested
without any producer dependency.

## Status & roadmap

**Version 0.1.0.** Built and tested:

- **P0** — Core SDK boundary (the five frozen contracts).
- **P1** — Parquet playback + headless PNG rendering.
- **P2** — Topology-aware layers (`HeatmapLayer`, `ClusterLayer`, `CentroidLayer`,
  `ShearLayer`), IDW interpolation, cluster-derived math (`normal_proxy`,
  centroid, shear); `render --view`.
- **P2.5** — Timeline sequence playback (`render-sequence`, `render-video`).
- **P3** — Live (push) visualization (`FrameFeed` abstraction, `LiveFrameSource`,
  `RollingNormalizer`, `live` CLI, built-in feeds).

**Not built** (one-way-dependency invariant preserved throughout):

- **P4** — Sync-grid synchronized playback + confidence overlay.
- **P5** — Tier-3 derived features reusing shared math (batch & live parity).
- **P6** — WebGL frontend implementing `Renderer`; video overlay + Tier-4 ML seams.

See `docs/ARCHITECTURE.md` for the full design.
