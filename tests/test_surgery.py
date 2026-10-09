from pathlib import Path

import pytest

from nasim.geometry import Position
from nasim.lattice_surgery import check_alignment, facing_edges
from nasim.model import Model
from nasim.placement import PlacedPatch, Placement
from nasim.surface_code import Edge, Patch

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


def test_horizontal_neighbours_face_right_left(model):
    placement = _side_by_side_placement(model, vertical=False)
    assert facing_edges(placement, 0, 1) == (Edge.RIGHT, Edge.LEFT)


def test_vertical_neighbours_face_top_bottom(model):
    placement = _side_by_side_placement(model, vertical=True)
    assert facing_edges(placement, 0, 1) == (Edge.TOP, Edge.BOTTOM)


def test_non_adjacent_patches_have_no_facing_edges(model):
    patch = Patch.rotated(3)
    placement = Placement(model=model, patches=(
        PlacedPatch(0, patch, Position(0.0, 0.0, 0.0)),
        PlacedPatch(1, patch, Position(500.0, 0.0, 0.0)),
    ))
    assert facing_edges(placement, 0, 1) is None


def test_check_merge_alignment_passes_for_adjacent_patches(model):
    placement = _side_by_side_placement(model, vertical=False)
    check_alignment(placement, 0, 1)  # should not raise


def test_check_merge_alignment_raises_for_non_adjacent_patches(model):
    patch = Patch.rotated(3)
    placement = Placement(model=model, patches=(
        PlacedPatch(0, patch, Position(0.0, 0.0, 0.0)),
        PlacedPatch(1, patch, Position(500.0, 0.0, 0.0)),
    ))
    with pytest.raises(ValueError, match="not next"):
        check_alignment(placement, 0, 1)