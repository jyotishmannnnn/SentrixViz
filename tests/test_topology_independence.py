"""Proves count / descriptor / topology independence: the SAME code builds a
RenderModel and a 2D layout for a 21-sensor glove, a 7-sensor tiny glove, and a
5-taxel non-glove finger module - no branch on count, name, or device class.
"""
from __future__ import annotations

import numpy as np

from sentrixviz.core.model import RenderModel


def test_model_builds_for_every_descriptor(descriptor_case):
    name, desc = descriptor_case
    model = RenderModel.from_descriptor(desc)
    # count independence: node count follows the descriptor, never a constant
    assert len(model.nodes) == len(desc.sensors)
    assert model.descriptor_version == desc.descriptor_version
    # edges are descriptor-derived (explicit or radius), undirected + unique
    for a, b in model.edges:
        assert a <= b
    assert len(set(model.edges)) == len(model.edges)


def test_projection_is_2d_and_count_agnostic(descriptor_case):
    name, desc = descriptor_case
    model = RenderModel.from_descriptor(desc)
    xy = model.project_2d()
    assert set(xy) == set(model.positioned_ids())
    for v in xy.values():
        assert v.shape == (2,)
        assert np.all(np.isfinite(v))


def test_no_finger_or_count_assumption(descriptor_case):
    """RobotFinger_v1 has no fingers and 5 sensors; it must build identically."""
    name, desc = descriptor_case
    model = RenderModel.from_descriptor(desc)
    if name == "RobotFinger_v1":
        assert model.device_class == "finger_module"
        assert len(model.nodes) == 5
        assert all(n.modality == "magnetic" for n in model.nodes)
