"""The five SDK contracts. Structural typing (typing.Protocol) so third parties
implement them without importing a base class.

    RenderModel  - static scene from a descriptor          (see core.model)
    FieldSet     - one frame's bound values                (see core.fields)
    FrameSource  - a timestamped stream of FieldSets        (this module)
    Renderer     - primitives -> pixels                     (this module)
    Layer        - (model, fieldset) -> visual primitives   (this module)

RenderModel and FieldSet are concrete dataclasses (their shape IS the contract).
FrameSource / Renderer / Layer are behavioural protocols with swappable impls.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

from .fields import FieldSet
from .model import RenderModel


@runtime_checkable
class FrameSource(Protocol):
    """A descriptor-paired, timestamped stream of frames.

    Unifies live and playback: a consumer never knows whether frames came from
    a socket or a parquet file. Phase 1 ships the playback (parquet) impl only;
    seek()/read() are reserved for Phase 4 synchronized playback and are NOT
    required of a Phase-1 source.

    Must know: its own descriptor identity, its timestamps, its channels.
    Must NEVER know: positions, topology, how anything is drawn.
    """
    @property
    def descriptor_version(self) -> str: ...
    @property
    def descriptor_hash(self) -> str | None: ...
    @property
    def ts_column(self) -> str: ...

    def channels(self) -> list[str]: ...
    def timestamps(self) -> list[int]: ...
    def __len__(self) -> int: ...
    def frame(self, index: int) -> FieldSet: ...


class Layer(Protocol):
    """Turns (model, fieldset) into renderer-agnostic visual primitives.

    A Layer outputs PHYSICAL scalars + geometry, never colors or pixels. The
    Renderer applies the colormap/normalizer. This keeps layers pure and lets
    one layer drive any renderer.

    Must NEVER import a rendering backend.
    """
    name: str

    def encode(self, model: RenderModel, fieldset: FieldSet | None) -> dict: ...


class Renderer(Protocol):
    """Consumes primitives from layers and produces an artifact (e.g. PNG).

    The only component allowed to touch a drawing backend. Swapping matplotlib
    for WebGL is replacing this and nothing else.
    """
    def render(
        self,
        model: RenderModel,
        layers: Sequence[Layer],
        fieldset: FieldSet | None = None,
        out: Path | None = None,
    ) -> bytes | Path: ...
