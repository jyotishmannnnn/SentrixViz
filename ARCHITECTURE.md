# SentrixViz Architecture

> Phase 0/1 scope. Records the SDK boundary and the descriptor→pixels data flow.

## ARCHITECTURE outline

1. **Position & posture** — beside the pipeline; depends on artifacts +
   `sentrix_contracts`; one-way, no producer/consumer imports.
2. **Data flow**
   ```
   Descriptor ──► RenderModel ──┐
   raw.parquet ─► FrameSource ──► FieldSet ──► Layer ──► primitives ──► Renderer ──► PNG
   ```
3. **The five contracts** — purpose, responsibilities, ownership, API surface,
   "must never know" (table mirrors CLAUDE.md; this file holds the detail).
4. **`TopologyDescriptor → RenderModel`** — `from_descriptor` reads
   positions/clusters/regions/frame; edges via `desc.neighbors()` (explicit or
   radius); no geometry computed in viz. 2D layout = PCA principal plane of
   positions (line/plane/patch, no special case). Cached per `descriptor_hash`.
5. **`raw.parquet → FieldSet`** — self-describing: `ts_column` +
   `sentrix_descriptor_*` from schema metadata; payload columns discovered via
   `parse_column`; `<sid>.valid` carried; NaN/invalid never interpolated.
6. **Layer/Renderer split** — layers emit physical scalars or category indices +
   geometry; the Renderer alone maps to color/pixels. Swapping renderers changes
   nothing upstream.
7. **Minimal viable renderer** — matplotlib headless PNG: smallest thing that
   exercises the whole boundary, CI-friendly, byte-assertable. Explicitly not the
   long-term frontend.
8. **Projection rationale** — why PCA-plane is the descriptor-driven default; the
   noted future gap (optional `layout` block for curved skins — additive,
   viz-only, defaults to PCA when absent).
9. **Canonical rendering model** — one scene (nodes + edges in device frame),
   many views; raster is a *view* (Phase 2), never the model.
10. **Testing strategy** — descriptor-parametrized; synth raw.parquet reproduces
    the producer at-rest contract without importing a producer.
11. **Phase map** — P0/P1/P2/P2.5/P3 built; P4–P6 noted with entry seams
    (`FrameFeed`, `Layer`, `Renderer`). P2 added topology-aware `Layer`s
    (heatmap via IDW; cluster/centroid/shear reductions) + three additive
    renderer primitives (`field`/`vectors`/`markers`) + `render --view`. P3
    added live (push) production: a single new abstraction `FrameFeed`
    (transport seam) + `LiveFrameSource` (iterable sibling of the parquet
    source) + `RollingNormalizer` (bounded adaptive policy in the existing seam)
    + `LiveSession` (latest-wins loop reusing the render path); no contract
    changed. Live data flow:
    ```
    raw bytes ─► FrameFeed ─► LiveFrameSource ─► FieldSet ─► Layer ─► Renderer ─► frame
                 (USB/socket/Capture-bus/Sim, never imported by viz)
    ```
12. **Open seams** — Sync-grid timeline / Sync-aligned feeds (P4), shared
    derived math reuse (P5), WebGL `Renderer` (P6). Hardware (RP2350), Sim
    streaming, and socket transports are each just a `FrameFeed` impl — no
    redesign; the live stack consumes any feed.
