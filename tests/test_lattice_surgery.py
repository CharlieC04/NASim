from pathlib import Path

import pytest

from nasim.geometry import Position
from nasim.lattice_surgery import check_alignment, facing_edges, merge_patches
from nasim.model import Model
from nasim.placement import PlacedPatch, Placement
from nasim.surface_code import AtomRole, Edge, Patch, _ancilla_weight_counts

PRESET = Path(__file__).resolve().parents[1] / "src" / "nasim" / "config" / "presets" / "2d.yaml"


@pytest.fixture
def model() -> Model:
    return Model.from_yaml(PRESET)


def _side_by_side_placement(model, vertical: bool = False) -> Placement:
    patch = Patch.rotated(3)
    unit = model.gate_pair_dist_um
    min_x, min_y, max_x, max_y = patch.local_bounds
    anchor0 = Position(-min_x * unit, -min_y * unit, 0.0)
    if vertical:
        h = (max_y - min_y) * unit
        anchor1 = Position(anchor0.x, anchor0.y + h, 0.0)
    else:
        w = (max_x - min_x) * unit
        anchor1 = Position(anchor0.x + w, anchor0.y, 0.0)
    return Placement(model=model, patches=(
        PlacedPatch(0, patch, anchor0),
        PlacedPatch(1, patch, anchor1),
    ))


def test_ancilla_weight_counts_for_standalone_d3_patch():
    patch = Patch.rotated(3)
    interior, boundary = _ancilla_weight_counts(patch)
    assert interior == 4
    assert boundary == 4


def test_ancilla_weight_counts_for_merged_6x3_patch():
    merged = Patch._build_rectangle(6, 3, code_distance=3)
    interior, boundary = _ancilla_weight_counts(merged)
    assert interior == 10
    assert boundary == 7
    assert interior + boundary == merged.num_ancilla


def test_merge_patches_horizontal_atom_counts(model):
    placement = _side_by_side_placement(model, vertical=False)
    merged, anchor, origin_qubit = merge_patches(placement, 0, 1)
    assert origin_qubit == 0
    data = [a for a in merged.atoms if a.role == AtomRole.DATA]
    assert len(data) == 18
    assert len(merged.atoms) == 35