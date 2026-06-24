"""sentrixviz CLI (Phase 1). Zero-dep argparse; three commands.

  sentrixviz inspect  RAW.parquet [--descriptor D] [--json]
        -> stdout summary: descriptor identity, sensor counts by modality,
           frames, duration, channels. No file artifact.

  sentrixviz topology --descriptor D [-o OUT.png]
        -> static geometry PNG (nodes by cluster + neighbor edges). Needs only a
           descriptor; no parquet. D is a bundled version name OR a JSON path.

  sentrixviz render   RAW.parquet [--descriptor D] [--frame N | --time US]
                      [--channel mag|bx|by|bz|ax|ay|az] [--modality M]
                      [--view raw|heatmap|clusters|centroid|shear] [-o OUT.png]
        -> one-frame PNG. --view selects the topology-aware layer stack
           (Phase 2); --view raw is the Phase-1 default.

  sentrixviz render-sequence RAW.parquet [--descriptor D] [same view/channel]
                      [--start N] [--stop N] [--step N] [-o OUT_DIR]
        -> ordered PNG sequence (000000.png, ...) + manifest.json over the whole
           timeline (Phase 2.5). Reuses the exact `render` path per frame.

  sentrixviz render-video RAW.parquet | --seq-dir DIR [--descriptor D] [view...]
                      [--fps F] [--format mp4|gif] [-o OUT]
        -> encode a PNG sequence into a video (Phase 2.5). Renders the sequence
           first unless --seq-dir points at one already. Optional ffmpeg (MP4),
           else Pillow (GIF); PNG sequence is always kept.

  sentrixviz live   RAW.parquet | --synth --descriptor D  [view/channel...]
                      [--max-frames N] [--fps F] [--alpha A]
                      [-o latest.png] [--out-dir DIR]
        -> render a live (push) stream (Phase 3). Replays a parquet AS a live
           stream, or runs a descriptor-driven synthetic feed (no hardware).
           `-o` is a latest-wins monitor; `--out-dir` writes numbered frames +
           manifest. Reuses the exact render path; live == another FrameSource.

`--descriptor` accepts a JSON path (synthetic / non-bundled revisions like
Mark2_tiny, RobotFinger_v1) or, for `topology`, a bundled version name.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sentrix_contracts import bundled_descriptor_path, load_descriptor

from .core.model import RenderModel
from .layers import (
    CentroidLayer,
    ClusterLayer,
    FieldLayer,
    HeatmapLayer,
    ShearLayer,
    TopologyLayer,
)
from .live import LiveSession
from .normalize import RollingNormalizer
from .render import MatplotlibRenderer
from .sources import (
    LiveFrameSource,
    ParquetFrameSource,
    ReplayFeed,
    SyntheticFeed,
    SyncFrameSource,
    resolve_descriptor,
)
from .timeline import Timeline, render_sequence, render_video


def _layers_for_view(view: str, channel: str, modality: str | None):
    """Map a --view name to a layer stack. `raw` reproduces the Phase-1
    behaviour exactly (backward compatible default)."""
    if view == "raw":
        return [FieldLayer(reduction=channel, modality=modality)]
    if view == "heatmap":
        return [HeatmapLayer(reduction=channel, modality=modality)]
    if view == "clusters":
        return [ClusterLayer(reduction=channel)]
    if view == "centroid":
        # contact centroid over the continuous field for context
        return [HeatmapLayer(reduction=channel, modality=modality),
                CentroidLayer(reduction=channel)]
    if view == "shear":
        return [ShearLayer(reduction=channel)]
    raise ValueError(f"unknown view: {view}")


def _load_descriptor_arg(arg: str):
    """A path -> load it; otherwise treat as a bundled version name."""
    p = Path(arg)
    if p.exists():
        return load_descriptor(p)
    return load_descriptor(bundled_descriptor_path(arg))


def _open_source(args):
    """Open the right FrameSource for a parquet artifact.

    Default: producer RAW (`ParquetFrameSource`). With `--silver`: the synced
    aligned Silver table (`SyncFrameSource`, VIZ-P4). A descriptor PATH (not a
    bundled name) is forwarded to the synced source for column inversion."""
    if getattr(args, "silver", False):
        dp = args.descriptor if (args.descriptor and Path(args.descriptor).exists()) else None
        return SyncFrameSource(args.raw, descriptor_path=dp)
    return ParquetFrameSource(args.raw)


def _cmd_inspect(args) -> int:
    src = _open_source(args)
    desc = resolve_descriptor(src, args.descriptor)
    ts = src.timestamps()
    info = {
        "artifact": str(src.path),
        "descriptor_version": desc.descriptor_version,
        "descriptor_hash": desc.descriptor_hash,
        "device_class": desc.device_class,
        "ts_column": src.ts_column,
        "n_sensors": len(desc.sensors),
        "n_magnetic": desc.n_mag,
        "n_dynamics": desc.n_dyn,
        "n_clusters": len(desc.clusters),
        "n_frames": len(src),
        "duration_us": (ts[-1] - ts[0]) if len(ts) > 1 else 0,
        "channels": src.channels(),
    }
    if args.json:
        print(json.dumps(info, indent=2))
    else:
        for k, v in info.items():
            print(f"{k:18} {v}")
    return 0


def _cmd_topology(args) -> int:
    desc = _load_descriptor_arg(args.descriptor)
    model = RenderModel.from_descriptor(desc)
    out = Path(args.out or f"{desc.descriptor_version}_topology.png")
    MatplotlibRenderer().render(model, [TopologyLayer()], fieldset=None, out=out)
    print(f"wrote {out}")
    return 0


def _cmd_render(args) -> int:
    src = _open_source(args)
    desc = resolve_descriptor(src, args.descriptor)
    model = RenderModel.from_descriptor(desc)
    if args.time is not None:
        index = src.index_at_time(args.time)
    else:
        index = args.frame
    fs = src.frame(index)
    suffix = "" if args.view == "raw" else f"_{args.view}"
    out = Path(args.out or f"{desc.descriptor_version}_frame{fs.frame_index}{suffix}.png")
    layers = _layers_for_view(args.view, args.channel, args.modality)
    MatplotlibRenderer().render(model, layers, fieldset=fs, out=out)
    print(f"wrote {out} (view {args.view}, frame {index}, t={fs.t_capture_us}us)")
    return 0


def _cmd_render_sequence(args) -> int:
    src = _open_source(args)
    desc = resolve_descriptor(src, args.descriptor)
    model = RenderModel.from_descriptor(desc)
    timeline = Timeline(src)
    suffix = "" if args.view == "raw" else f"_{args.view}"
    out_dir = Path(args.out or f"{desc.descriptor_version}_seq{suffix}")
    layers = _layers_for_view(args.view, args.channel, args.modality)
    manifest = render_sequence(
        model, layers, timeline, out_dir, MatplotlibRenderer(),
        start=args.start, stop=args.stop, step=args.step,
        view=args.view, channel=args.channel,
    )
    print(
        f"wrote {manifest['n_frames']} frames to {out_dir}/ "
        f"(view {args.view}, ~{manifest['fps_estimate']:.2f} fps, "
        f"{len(manifest['dropped'])} gap(s))"
    )
    return 0


def _cmd_render_video(args) -> int:
    if args.seq_dir:
        seq_dir = Path(args.seq_dir)
        fps = args.fps if args.fps is not None else 30.0
        name_width = 6
    else:
        if not args.raw:
            print("render-video: pass RAW.parquet or --seq-dir", file=sys.stderr)
            return 2
        src = _open_source(args)
        desc = resolve_descriptor(src, args.descriptor)
        model = RenderModel.from_descriptor(desc)
        timeline = Timeline(src)
        suffix = "" if args.view == "raw" else f"_{args.view}"
        seq_dir = Path(f"{desc.descriptor_version}_seq{suffix}")
        layers = _layers_for_view(args.view, args.channel, args.modality)
        manifest = render_sequence(
            model, layers, timeline, seq_dir, MatplotlibRenderer(),
            step=args.step, view=args.view, channel=args.channel,
        )
        name_width = manifest["name_width"]
        # default playback rate = the dataset's own cadence when not overridden
        fps = args.fps if args.fps is not None else (manifest["fps_estimate"] or 30.0)
    out = Path(args.out or seq_dir.with_suffix(f".{args.format}").name)
    written = render_video(seq_dir, out, fps=fps, fmt=args.format, name_width=name_width)
    print(f"wrote {written} ({fps:.2f} fps) from {seq_dir}/")
    return 0


def _cmd_live(args) -> int:
    # descriptor discovery: replay -> parquet metadata; synth -> --descriptor.
    if args.synth:
        if not args.descriptor:
            print("live --synth: pass --descriptor", file=sys.stderr)
            return 2
        desc = _load_descriptor_arg(args.descriptor)
        feed = SyntheticFeed(desc, n_frames=args.max_frames)
    else:
        if not args.raw:
            print("live: pass RAW.parquet to replay, or --synth --descriptor D",
                  file=sys.stderr)
            return 2
        src = ParquetFrameSource(args.raw)
        desc = resolve_descriptor(src, args.descriptor)
        feed = ReplayFeed(src, loop=args.loop)

    model = RenderModel.from_descriptor(desc)
    live = LiveFrameSource(feed)
    layers = _layers_for_view(args.view, args.channel, args.modality)

    out = out_dir = None
    if args.out_dir:
        out_dir = Path(args.out_dir)
    if args.out or not args.out_dir:
        # default to a latest-wins monitor when nothing else is requested
        suffix = "" if args.view == "raw" else f"_{args.view}"
        out = Path(args.out or f"{desc.descriptor_version}_live{suffix}.png")

    session = LiveSession(
        model, layers, MatplotlibRenderer(),
        normalizer=RollingNormalizer(alpha=args.alpha),
        view=args.view, channel=args.channel,
    )
    n = [0]

    def _tick(rec):
        n[0] = rec["emit"] + 1

    manifest = session.run(
        live, out=out, out_dir=out_dir,
        max_frames=args.max_frames, fps=args.fps, on_frame=_tick,
    )
    where = out_dir if out_dir else out
    print(
        f"live: rendered {manifest['n_frames']} frame(s) -> {where} "
        f"(view {args.view}, alpha {args.alpha}, {manifest['dropped']} dropped)"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sentrixviz")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("inspect", help="summarize a raw.parquet artifact")
    pi.add_argument("raw")
    pi.add_argument("--descriptor", default=None)
    pi.add_argument("--silver", action="store_true",
                    help="read a synced/aligned Silver table (SyncFrameSource) instead of raw")
    pi.add_argument("--json", action="store_true")
    pi.set_defaults(func=_cmd_inspect)

    pt = sub.add_parser("topology", help="render static descriptor geometry")
    pt.add_argument("--descriptor", required=True, help="bundled version name or JSON path")
    pt.add_argument("-o", "--out", default=None)
    pt.set_defaults(func=_cmd_topology)

    pr = sub.add_parser("render", help="render one frame's field")
    pr.add_argument("raw")
    pr.add_argument("--descriptor", default=None)
    pr.add_argument("--silver", action="store_true",
                    help="render the synced/aligned Silver table (SyncFrameSource, VIZ-P4)")
    g = pr.add_mutually_exclusive_group()
    g.add_argument("--frame", type=int, default=0)
    g.add_argument("--time", type=int, default=None, help="t_capture_us; nearest frame")
    pr.add_argument("--channel", default="mag")
    pr.add_argument("--modality", default=None)
    pr.add_argument("--view", default="raw",
                    choices=["raw", "heatmap", "clusters", "centroid", "shear"],
                    help="visualization mode (default raw == Phase-1 behaviour)")
    pr.add_argument("-o", "--out", default=None)
    pr.set_defaults(func=_cmd_render)

    def _add_view_args(sp):
        sp.add_argument("--descriptor", default=None)
        sp.add_argument("--silver", action="store_true",
                        help="read a synced/aligned Silver table (SyncFrameSource, VIZ-P4)")
        sp.add_argument("--channel", default="mag")
        sp.add_argument("--modality", default=None)
        sp.add_argument("--view", default="raw",
                        choices=["raw", "heatmap", "clusters", "centroid", "shear"])

    ps = sub.add_parser("render-sequence", help="render the whole timeline to a PNG sequence")
    ps.add_argument("raw")
    _add_view_args(ps)
    ps.add_argument("--start", type=int, default=0)
    ps.add_argument("--stop", type=int, default=None)
    ps.add_argument("--step", type=int, default=1, help="frame stride (decimation)")
    ps.add_argument("-o", "--out", default=None, help="output directory")
    ps.set_defaults(func=_cmd_render_sequence)

    pv = sub.add_parser("render-video", help="encode a PNG sequence into a video")
    pv.add_argument("raw", nargs="?", default=None,
                    help="raw.parquet to render first (omit if using --seq-dir)")
    pv.add_argument("--seq-dir", default=None, help="use an existing PNG sequence dir")
    _add_view_args(pv)
    pv.add_argument("--step", type=int, default=1)
    pv.add_argument("--fps", type=float, default=None,
                    help="playback fps (default: dataset cadence)")
    pv.add_argument("--format", default="mp4", choices=["mp4", "gif"])
    pv.add_argument("-o", "--out", default=None)
    pv.set_defaults(func=_cmd_render_video)

    pl = sub.add_parser(
        "live",
        help="render a live (push) stream: replay a parquet or a synthetic feed",
    )
    pl.add_argument("raw", nargs="?", default=None,
                    help="raw.parquet to replay as a live stream (omit with --synth)")
    pl.add_argument("--synth", action="store_true",
                    help="descriptor-driven synthetic stream (no parquet, no hardware)")
    pl.add_argument("--loop", action="store_true", help="replay: loop forever")
    _add_view_args(pl)
    pl.add_argument("--max-frames", type=int, default=16,
                    help="stop after N frames (synth default length too)")
    pl.add_argument("--fps", type=float, default=None,
                    help="throttle to this cadence (default: as fast as frames arrive)")
    pl.add_argument("--alpha", type=float, default=0.1,
                    help="RollingNormalizer relax rate (smaller = steadier)")
    pl.add_argument("-o", "--out", default=None,
                    help="latest-wins monitor PNG (overwritten each frame)")
    pl.add_argument("--out-dir", default=None,
                    help="write numbered frames + manifest.json instead/also")
    pl.set_defaults(func=_cmd_live)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
