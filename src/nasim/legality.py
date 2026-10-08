from __future__ import annotations

from nasim.move import Move, compatible_2d
from nasim.geometry import Position, distance
from nasim.model import Model

def check_frame_no_cross(frame: list[Move]) -> None:
    for i in range(len(frame)):
        for j in range(i+1, len(frame)):
            if not compatible_2d(frame[i], frame[j]):
                raise ValueError("no-crossing violation")

def check_schedule_no_cross(frames: list[list[Move]]) -> None:
    for frame in frames:
        check_frame_no_cross(frame)

def check_blockade(pos_a: Position, pos_b: Position, model: Model) -> None:

    d = distance(pos_a, pos_b)
    if d > model.blockade_radius_um:
        raise ValueError("Patch distance exceeds blockade radius")