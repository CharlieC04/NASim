import pytest

from pathlib import Path

from nasim.geometry import Position
from nasim.legality import check_frame_no_cross, check_schedule_no_cross, check_blockade
from nasim.move import Move
from nasim.model import Model

PRESET = Path(__file__).resolve().parents[1] / "src" / "nasim" / "config" / "presets" / "2d.yaml"


def test_passes_a_legal_frame():
    frame = [
        Move(0, Position(0, 0, 0), Position(10, 0, 0)),
        Move(1, Position(5, 10, 0), Position(15, 20, 0)),
    ]
    check_frame_no_cross(frame)  # should not raise


def test_raises_on_crossing_moves_in_the_same_frame():
    frame = [
        Move(0, Position(0, 0, 0), Position(10, 0, 0)),
        Move(1, Position(5, 0, 0), Position(2, 0, 0)),
    ]
    with pytest.raises(ValueError, match="no-crossing violation"):
        check_frame_no_cross(frame)


def test_check_schedule_no_crossing_checks_every_frame():
    frames = [
        [Move(0, Position(0, 0, 0), Position(10, 0, 0))],
        [
            Move(1, Position(5, 0, 0), Position(2, 0, 0)),
            Move(2, Position(5, 0, 0), Position(8, 0, 0)),
        ],
    ]
    with pytest.raises(ValueError):
        check_schedule_no_cross(frames)

def test_blockade_legal_within_radius():
    model = Model.from_yaml(PRESET)
    check_blockade(Position(0, 0, 0), Position(model.blockade_radius_um, 0, 0), model)  # == radius, legal


def test_blockade_illegal_beyond_radius():
    model = Model.from_yaml(PRESET)
    with pytest.raises(ValueError, match="distance exceeds"):
        check_blockade(Position(0, 0, 0), Position(model.blockade_radius_um + 0.1, 0, 0), model)