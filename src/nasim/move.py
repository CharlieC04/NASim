from __future__ import annotations

import math
from dataclasses import dataclass

from nasim.geometry import Position
from nasim.model import Model
from nasim.placement import Placement

@dataclass(frozen=True)
class Move:

    qubit: int
    source: Position
    target: Position
    clearance_um: float = 0.0

    def distance_um(self) -> float:
        return math.hypot(self.target.x - self.source.x, self.target.y - self.source.y)

def move_duration_us(move: Move, model: Model) -> float:
    
    """
    Min-jerk movement from Piqasso
    """

    return (15/8) * (move.distance_um() / model.v_xy_um_per_us)

def conflict_reason(a: Move, b: Move) -> tuple[str, str] | None:

    """Per-axis AOD conflict between two moves, or None if compatible on
    both axes: "shared_source" (same start, diverging ends), 
    "shared_target" (same end, diverging starts), or
    "crossing" (trajectories overlap in between)."""

    checks = (
        (a.source.x, a.target.x, b.source.x, b.target.x, "x"),
        (a.source.y, a.target.y, b.source.y, b.target.y, "y")
    )

    for a0, a1, b0, b1, axis in checks:
        if a0 == b0 and a1 != b1: return (axis, "shared_source")
        if a1 == b1 and a0 != b0: return (axis, "shared_target")
        if a0 < b0 and a1 >= b1: return (axis, "crossing")
        if a0 > b0 and a1 <= b1: return (axis, "crossing")

def compatible_2d(a: Move, b: Move) -> bool:

    if conflict_reason(a, b) is not None:
        return False

    margin = a.clearance_um + b.clearance_um
    if margin > 0:
        d_source = math.hypot(a.source.x - b.source.x, a.source.y - b.source.y)
        d_target = math.hypot(a.target.x - b.target.x, a.target.y - b.target.y)
        if d_source < margin or d_target < margin: return False

    return True

def _greedy_max_indep_set(moves: list[Move]) -> list[Move]:

    """
    Build conflict graph and pick in asc order
    """

    n = len(moves)
    conflict: dict[int, set[int]] = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i+1, n):
            if not compatible_2d(moves[i], moves[j]):
                conflict[i].add(j)
                conflict[j].add(i)

    order = sorted(range(n), key=lambda i: len(conflict[i]))
    chosen: list[int] = []
    visited: set[int] = set()
    for i in order:
        if i not in visited:
            chosen.append(i)
            visited.add(i)
            visited.update(conflict[i])

    return [moves[i] for i in chosen]

def _park(move: Move, axis: str, other: Move) -> Move:

    """
    Short orthog move to clear other conflict
    """

    margin = move.clearance_um + other.clearance_um
    step = margin if margin > 0 else 1.0
    if axis == "x":
        forward = move.target.x >= move.source.x
        parked = Position(move.source.x + (step if forward else -step), move.source.y, move.source.z)
    else:
        forward = move.target.y >= move.source.y
        parked = Position(move.source.x, move.source.y + (step if forward else -step), move.source.z)

    return Move(qubit=move.qubit, source=move.source, target=parked, clearance_um=move.clearance_um)

def _remainder(move: Move, parking: Move) -> Move:
    return Move(qubit=move.qubit, source=parking.target, target=move.target, clearance_um=move.clearance_um)

def _aod_violations(frame_moves: list[Move], occupied: set[tuple[float, float]]) -> list[tuple[float, float]]:

    xs = {m.source.x for m in frame_moves}
    ys = {m.source.y for m in frame_moves}
    intended = {(m.source.x, m.source.y) for m in frame_moves}

    return [
        (x, y)
        for x in xs for y in ys
        if (x, y) not in intended and (x, y) in occupied
    ]

def _resolve_violations(moves: list[Move], occupied: set[tuple[float, float]]) -> tuple[list[Move], list[Move]]:

    """
    Remove whichever move clears most violations
    """

    remaining = list(moves)
    removed: list[Move] = []

    while True:
        violations = _aod_violations(remaining, occupied)
        if not violations:
            return remaining, removed

        best_qubit, best_cleared = None, -1
        for m in remaining:
            trial = [mv for mv in remaining if mv.qubit != m.qubit]
            cleared = len(violations) - len(_aod_violations(trial, occupied))
            if cleared > best_cleared:
                best_cleared, best_qubit = cleared, m.qubit

        removed.append(next(m for m in remaining if m.qubit == best_qubit))
        remaining = [m for m in remaining if m.qubit != best_qubit]

def _build_frame_with_parking(moves: list[Move], occupied: set[tuple[float, float]]) -> tuple[list[Move], list[Move]]:

    """
    Build a frame of concurrent moves
    """

    chosen = _greedy_max_indep_set(moves)
    chosen_by_qubit = {m.qubit: m for m in chosen}
    deferred = [m for m in moves if m.qubit not in chosen_by_qubit]
    parking_moves: list[Move] = []

    progress = True
    while progress and deferred:
        progress = False
        still_deferred = []

        for move in deferred:
            blockers = [
                (k, reason) for k in chosen_by_qubit.values()
                if (reason := conflict_reason(move, k)) is not None
            ]

            if len(blockers) == 1 and blockers[0][1][1] == "shared_source":
                partner, (axis, _) = blockers[0]
                parking = _park(move, axis, partner)
                remainder = _remainder(move, parking)
                if all(compatible_2d(remainder, k) for k in chosen_by_qubit.values()) and all(compatible_2d(parking, p) for p in parking_moves):
                    parking_moves.append(parking)
                    chosen_by_qubit[remainder.qubit] = remainder
                    progress = True
                    continue

            still_deferred.append(move)

        deferred = still_deferred

    final_moves, aod_deferred = _resolve_violations(list(chosen_by_qubit.values()), occupied)
    if aod_deferred:
        dropped_qubits = {m.qubit for m in aod_deferred}
        parking_moves = [p for p in parking_moves if p.qubit not in dropped_qubits]

    return parking_moves, final_moves

def legalise_frames(moves: list[Move], stationary: list[Position] = ()) -> list[list[Move]]:

    """
    Partition moves into AOD frames. Repeatedly extract one maximal set of mutually compatible moves
    as a frame until all are scheduled
    """

    occupied = {(p.x, p.y) for p in stationary}
    remaining = list(moves)
    frames: list[list[Move]] = []
    while remaining:
        pending_sources = {m.source for m in remaining}
        safe = [m for m in remaining if m.target not in pending_sources]
        if not safe:
            cycle_qubits = sorted(m.qubit for m in remaining)
            raise ValueError("Cannot make progress due to a cycle")

        parking_moves, frame = _build_frame_with_parking(safe, occupied)
        if parking_moves:
            frames.append(parking_moves)
        scheduled = {m.qubit for m in frame}
        frames.append(frame)
        remaining = [m for m in remaining if m.qubit not in scheduled]

    return frames

def moves_between(before: Placement, after: Placement) -> list[Move]:

    """
    Straight-line Move for every qubit whose anchor changes
    """

    before_by_qubit = {p.qubit: p for p in before.patches}
    moves = []
    for placed in after.patches:
        prior = before_by_qubit[placed.qubit]
        if prior.anchor != placed.anchor:
            radius = placed.patch.radius_um(after.model.gate_pair_dist_um)
            moves.append(
                Move(
                    qubit=placed.qubit,
                    source=prior.anchor,
                    target=placed.anchor,
                    clearance_um=radius
                )
            )

    return moves