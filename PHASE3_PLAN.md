# SentrixViz Phase 3 — Live Visualization

> Goal: `raw bytes → Capture → FrameSource → Layers → Renderer → live view`,
> preserving descriptor / topology / count independence. Live must feel like
> **another FrameSource**, not a special subsystem. No frozen contract changes.

## 1. Architecture review — the smallest live path

The frozen `FrameSource` protocol is **finite and random-access**
(`__len__`, `frame(i)`, `timestamps()`). A live stream is **unbounded and
push-based**: it has no length and no past. Forcing it through the random-access
half would be a lie. But hard rule #7 already says *"live vs playback is a
FrameSource distinction, not a renderer one — both emit `(timestamp, FieldSet)`"*.

The only thing every downstream component (`Layer` → `Renderer` →
`NormalizationPolicy`) actually consumes is a **`FieldSet`**. So Phase 3 adds a
push-mode *sibling* producer that emits `FieldSet`s, and reuses the entire
descriptor→pixels stack verbatim:

```
raw bytes → [transport] → FrameFeed → LiveFrameSource → FieldSet
            → existing Layers → existing Renderer (+ RollingNormalizer) → frame
```

Reused unchanged: `RenderModel`, `FieldSet`, `Layer`, `Renderer`,
`NormalizationPolicy` seam, every Phase-2 layer, the matplotlib renderer.
New abstractions: exactly one — `FrameFeed` (the transport seam). Everything
else is a concrete impl of an existing protocol or a thin loop.

## 2. Live data source — transport tradeoffs

| Option | Couples Viz to | Verdict |
|---|---|---|
| Direct USB | RP2350 driver, libusb | ✗ hardware coupling, untestable |
| Capture integration (import) | SentrixCapture code | ✗ violates hard rule #3 |
| **Frame bus / queue (push)** | an iterator of raw frames | ✓ chosen |
| Socket | wire format only | ✓ a `FrameFeed` impl |

Decision: Viz couples only to **`FrameFeed`** — an iterator yielding `RawFrame`
records (the at-rest column convention + ts + frame_index). The transport
(USB reader, socket, in-process Capture bus, Sim stream) lives *behind* the
feed and is never imported by Viz. This is identical in spirit to how
`ParquetFrameSource` hides pyarrow behind the source.

`QueueFeed` is the canonical live tap: **read-only, latest-wins, never
backpressures the producer** (CLAUDE.md P3 note). A bounded ring drops the
oldest frame instead of blocking the recorder.

## 3. `LiveFrameSource`

Same philosophy as `ParquetFrameSource` (descriptor-paired, emits
`(timestamp, FieldSet)`, backend-blind) but **iterable, not indexable**. Wraps
a `FrameFeed`, binds each `RawFrame` via the existing `bind_row`. Exposes
`descriptor_version` / `descriptor_hash` / `ts_column` so `resolve_descriptor`
works exactly as for parquet. Hardware-independent (any feed) and
Sim-streaming-compatible (a Sim feed) by construction.

## 4. `RollingNormalizer`

Live cannot use `GlobalNormalizer` (no whole-stream scan). Plugs into the same
`NormalizationPolicy` seam:

- **Bounded memory** — O(keys), independent of stream length (one `[lo,hi]` per
  key, no frame buffer).
- **Stable visuals** — EMA smoothing damps per-frame jitter → no flicker.
- **Gradual adaptation** — auto-gain: expand the range *immediately* when a
  frame exceeds it (never clip a contact peak), relax *slowly* via EMA (range
  never collapses on a quiet frame). `alpha` controls the relax rate.
- Works for every channel the seam exposes: `color_range` (heatmap / centroid /
  points), `size_range` (markers), `vector_scale` (shear — primed once with the
  static anchor span, since positions don't move), `categories` (clusters —
  union of labels seen).

## 5. Live CLI — `sentrixviz live`

```
sentrixviz live [RAW.parquet] [--synth] [--descriptor D]
                [--view raw|heatmap|clusters|centroid|shear] [--channel C] [--modality M]
                [--max-frames N] [--fps F] [--alpha A]
                [--out latest.png] [--out-dir DIR]
```

- **inputs**: a `RAW.parquet` replayed as a live stream (no hardware), or
  `--synth` (descriptor-driven synthetic stream). Socket/USB feeds are noted
  future feeds (same CLI shape).
- **outputs**: `--out latest.png` is a latest-wins monitor (overwritten each
  frame); `--out-dir DIR` writes numbered frames + a `manifest.json` (the
  video-overlay / sync key, same schema as `render-sequence`).
- **update cadence**: `--fps` throttles; default follows the stream.
- **descriptor discovery**: replay → parquet metadata (`resolve_descriptor`);
  synth → `--descriptor`. Identical to every other command.

## 6. Future compatibility (no redesign)

- **RP2350 hardware** → a `UsbFeed` emitting `RawFrame`s; nothing else changes.
- **Sim live streaming** → a `SimFeed`; same.
- **Sync-aligned streams** → a feed that as-of-joins many feeds into one
  `RawFrame` stream (Phase 4 lives entirely inside the feed).
- **Video overlays** → the live manifest already records `t_capture_us` per
  frame (the sync key).
- **WebGL renderer** → swap the `Renderer`; `LiveFrameSource` emits the same
  `FieldSet`s and layers emit the same primitives.

## 7. Testing (synthetic, no hardware)

- descriptor / topology / count independence — drive `LiveSession` with
  `SyntheticFeed` across Mark2_v1 (21+3) / Mark2_tiny (7) / RobotFinger_v1 (5).
- rolling normalization stability — step/peak/quiet streams: range converges,
  adapts gradually on relax, never NaN, **memory bounded** (state size constant
  as frames grow).
- live stream robustness — dropouts (NaN frames) render hollow not crash;
  `QueueFeed` drops oldest under backpressure and never blocks the producer;
  `ReplayFeed → LiveFrameSource` yields `FieldSet`s **byte-equal** to
  `ParquetFrameSource.frame(i)` (live path parity).

## 8. Acceptance criteria

1. `sentrixviz live RAW.parquet --view heatmap --max-frames 8 --out-dir D`
   writes 8 frames + manifest, for any bundled/synthetic descriptor.
2. `sentrixviz live --synth --descriptor RobotFinger_v1 --view shear` runs with
   no parquet and no hardware.
3. Replay parity test green (live `FieldSet` == playback `FieldSet`).
4. Rolling stability + bounded-memory tests green.
5. No frozen contract touched; full existing suite (136 tests) still green.
</content>
</invoke>
