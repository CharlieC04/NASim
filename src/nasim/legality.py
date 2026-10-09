from __future__ import annotations

from nasim.move import Move, compatible_2d
from nasim.geometry import Position, distance
from nasim.model import Model

def check_frame_no_cross(frame: list[Move]) -> None:

    """Raise if any two moves in the frame are AOD-incompatible."""

    for i in range(len(frame)):
        for j in range(i+1, len(frame)):
            if not compatible_2d(frame[i], frame[j]):
                raise ValueError("no-crossing violation")

def check_schedule_no_cross(frames: list[list[Move]]) -> None:

    """Raise on the first frame that fails check_frame_no_cross."""

    for frame in frames:
        check_frame_no_cross(frame)

def check_blockade(pos_a: Position, pos_b: Position, model: Model) -> None:

    """Raise if two atoms are farther apart than the Rydberg blockade
    radius
    """

    d = distance(pos_a, pos_b)
    if d > model.blockade_radius_um:
        raise ValueError("Patch distance exceeds blockade radius")