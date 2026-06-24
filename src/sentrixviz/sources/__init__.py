"""Frame sources - adapters that feed FieldSets to the SDK.

Phase 1: playback from raw.parquet.
Phase 3 (BUILT): live (push) production - LiveFrameSource over a FrameFeed; see
  live.py. Built-in feeds: ReplayFeed, SyntheticFeed, QueueFeed (no hardware).
Phase 4 (noted, not built): a Sync-grid timeline that as-of joins many sources.
"""
from __future__ import annotations

from pathlib import Path

from sentrix_contracts import Descriptor, bundled_descriptor_path, load_descriptor

from .live import (
    FrameFeed,
    LiveFrameSource,
    QueueFeed,
    RawFrame,
    ReplayFeed,
    SyntheticFeed,
)
from .parquet import ParquetFrameSource
from .sync import SyncFrameSource


def resolve_descriptor(
    source,
    descriptor_path: str | Path | None = None,
) -> Descriptor:
    """Pair a source (playback OR live) with its descriptor.

    Works on any source exposing descriptor_version / descriptor_hash (a
    ParquetFrameSource or a LiveFrameSource), so the live path resolves its
    descriptor identically to playback.

    Resolution order:
      1. explicit `descriptor_path` (for synthetic / non-bundled revisions),
      2. the bundled descriptor named by the artifact's metadata.
    When the artifact carries a descriptor_hash, it is checked against the
    resolved descriptor and a mismatch is a hard error (provenance closure).
    """
    if descriptor_path is not None:
        desc = load_descriptor(descriptor_path)
    else:
        ver = source.descriptor_version
        if not ver:
            where = getattr(source, "path", "<stream>")
            raise ValueError(
                f"{where}: no descriptor_version in metadata; "
                f"pass --descriptor explicitly."
            )
        desc = load_descriptor(bundled_descriptor_path(ver))

    want = source.descriptor_hash
    if want and desc.descriptor_hash and want != desc.descriptor_hash:
        raise ValueError(
            f"descriptor hash mismatch: artifact={want} descriptor={desc.descriptor_hash}"
        )
    return desc


__all__ = [
    "ParquetFrameSource", "SyncFrameSource", "resolve_descriptor",
    "LiveFrameSource", "FrameFeed", "RawFrame",
    "ReplayFeed", "SyntheticFeed", "QueueFeed",
]
