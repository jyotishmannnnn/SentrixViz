"""render_video - an OPTIONAL thin encode step over a rendered PNG sequence.

Posture: the PNG sequence is the canonical, always-works deliverable. Video is
a convenience built on top of it, so there is no hard ffmpeg dependency and no
new rendering path. The PNG directory IS the frame cache - encoding reads the
files render_sequence already wrote, so re-encoding (MP4 then GIF) never
re-renders.

Backend selection (simplest thing that works on the host):
  - MP4 requested AND ffmpeg on PATH  -> ffmpeg H.264.
  - GIF requested (or MP4 with no ffmpeg) AND Pillow present -> Pillow GIF.
  - neither available -> raise, leaving the PNG sequence intact.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def _png_files(seq_dir: Path) -> list[Path]:
    files = sorted(seq_dir.glob("[0-9]*.png"))
    if not files:
        raise FileNotFoundError(f"no numbered PNG frames in {seq_dir}")
    return files


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _encode_mp4_ffmpeg(seq_dir: Path, out: Path, fps: float, name_width: int) -> Path:
    pattern = str(seq_dir / f"%0{name_width}d.png")
    cmd = [
        "ffmpeg", "-y",
        "-framerate", f"{fps:.6f}",
        "-i", pattern,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        # pad odd dimensions; H.264 needs even width/height
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out


def _encode_gif_pillow(seq_dir: Path, out: Path, fps: float) -> Path:
    from PIL import Image  # matplotlib already pulls Pillow in practice

    files = _png_files(seq_dir)
    frames = [Image.open(f).convert("RGB") for f in files]
    duration_ms = int(1000.0 / fps) if fps > 0 else 100
    frames[0].save(
        out, save_all=True, append_images=frames[1:],
        duration=duration_ms, loop=0, optimize=True,
    )
    return out


def render_video(
    seq_dir: str | Path,
    out: str | Path,
    *,
    fps: float = 30.0,
    fmt: str | None = None,
    name_width: int = 6,
) -> Path:
    """Encode the PNG sequence in `seq_dir` into a single video at `out`.

    `fmt` is inferred from `out`'s suffix when None ('.mp4' or '.gif'). MP4
    falls back to GIF when ffmpeg is absent (the suffix of the returned path
    reflects what was actually written).
    """
    seq_dir = Path(seq_dir)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    _png_files(seq_dir)  # fail fast if nothing to encode

    fmt = (fmt or out.suffix.lstrip(".") or "mp4").lower()

    if fmt == "mp4":
        if _ffmpeg_available():
            return _encode_mp4_ffmpeg(seq_dir, out, fps, name_width)
        # graceful downgrade rather than failing the whole export
        out = out.with_suffix(".gif")
        return _encode_gif_pillow(seq_dir, out, fps)
    if fmt == "gif":
        return _encode_gif_pillow(seq_dir, out, fps)
    raise ValueError(f"unsupported video format: {fmt!r} (use mp4 or gif)")
