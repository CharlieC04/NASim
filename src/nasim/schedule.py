from __future__ import annotations

import math
from dataclasses import dataclass, field

from nasim.circuit import Circuit, GateType
from nasim.model import Model
from nasim.move import Move, move_duration_us, legalise_frames, moves_between, conflict_reason
from nasim.placement import Placement, PlacedPatch
from nasim.geometry import Position
from nasim.surface_code import AtomRole, Patch, _ancilla_weight_counts, Edge
from nasim.lattice_surgery import merge_patches

AtomId = tuple[int, tuple[int, int]] # (qubit, local coord)

def _all_atom_ids(placement: Placement) -> list[AtomId]:
    return [(p.qubit, atom.local) for p in placement.patches for atom in p.patch.atoms]

def _data_atom_ids(placement: Placement, qubit: int) -> list[AtomId]:
    placed = next(p for p in placement.patches if p.qubit == qubit)
    return [(qubit, atom.local) for atom in placed.patch.atoms if atom.role == AtomRole.DATA]

def _patch_atom_ids(placement: Placement, qubit: int) -> list[AtomId]:
    placed = next(p for p in placement.patches if p.qubit == qubit)
    return [(qubit, atom.local) for atom in placed.patch.atoms]

@dataclass
class ScheduleOp:
    kind: str
    start_us: float
    duration_us: float
    targets: tuple[AtomId, ...]
    detail: dict

@dataclass
class Schedule:

    """
    Accumulate times sequence of operations:
    - Ng1: physical 1q gate count (one per addressed atom per 1Q stage)
    - Nsp: carrier-background exposures (every other live atom, once per 1Q gate stage's global pulse) 
    - Ng2: physical 2q gate count
    - Nh: handover count (PICK+DROP per atom per move)
    - idle_us: cum idle time per atom
    """

    model: Model
    all_atoms: set[AtomId]
    clock_us: float = 0.0
    ops: list[ScheduleOp] = field(default_factory=list)

    Ng1: int = 0
    Nsp: int = 0
    Ng2: int = 0
    Nh: int = 0
    Nmeas: int = 0

    idle_us: dict[AtomId, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.all_atoms = set(self.all_atoms)
        for atom in self.all_atoms:
            self.idle_us.setdefault(atom, 0.0)

    def _advance(self, targets: list[AtomId], duration_us: float) -> float:
        start_us = self.clock_us
        target_set = set(targets)
        for atom in self.all_atoms:
            if atom not in target_set:
                self.idle_us[atom] += duration_us

        self.clock_us = start_us + duration_us
        return start_us

    def retarget_atoms(self, remove: set[AtomId], add: set[AtomId]) -> None:

        for atom in remove:
            self.all_atoms.discard(atom)
        for atom in add:
            self.all_atoms.add(atom)
            self.idle_us.setdefault(atom, 0.0)

    def record_1q_stage(self, circuit: Circuit, stage_idx: int, placement: Placement) -> None:

        """
        Execute circuit.stages[stage_idx] as one shared global pulse (1Q gates only)
        """

        stage = circuit.stages[stage_idx]
        if any(g.type is GateType.TWO_QUBIT for g in stage):
            raise NotImplementedError(f"stage {stage_idx} contains 2Q gate")

        if not stage: return

        targets: list[AtomId] = []
        for gate in stage:
            (qubit,) = gate.qubits
            targets.extend(_data_atom_ids(placement, qubit))

        duration = self.model.single_qubit_duration_us
        start_us = self._advance(targets, duration)

        num_live = len(self.all_atoms)
        self.Ng1 += len(targets)
        self.Nsp += num_live - len(targets)

        self.ops.append(ScheduleOp(
            kind="1q_stage", start_us=start_us, duration_us=duration,
            targets=tuple(targets),
            detail={"stage": stage_idx, "qubits": [g.qubits[0] for g in stage]}
        ))

    def record_move_frame(self, frame: list[Move], placement_before: Placement) -> None:

        """
        Execute one frame of concurrent moves. Duration is slowest move + handover
        """

        if not frame: return
        transit = max(move_duration_us(m, self.model) for m in frame)
        handover = self.model.handover_time_us
        duration = 2 * handover + transit

        targets: list[AtomId] = []
        for move in frame:
            atoms = _patch_atom_ids(placement_before, move.qubit)
            targets.extend(atoms)
            self.Nh += 2 * len(atoms)

        start_us = self._advance(targets, duration)

        self.ops.append(ScheduleOp(
            kind="move_frame", start_us=start_us, duration_us=duration,
            targets=tuple(targets),
            detail={"qubits": [m.qubit for m in frame], "moves": tuple(frame)}
        ))

    def _record_syndrome_round(self, merged_patch: Patch, targets: list[AtomId]) -> None:

        """
        Round = 4 sequential CZ pulses (naive)
        """

        interior_count, boundary_count = _ancilla_weight_counts(merged_patch)
        cz_duration = self.model.cz_duration_us

        for step in range(4):
            start_us = self._advance(targets, cz_duration)
            active = interior_count + (boundary_count if step < 2 else 0)
            self.Ng2 += active
            self.ops.append(ScheduleOp(
                kind="2q_substep", start_us=start_us, duration_us=cz_duration,
                targets=tuple(targets), detail={"substep": step, "active_gates": active}
            ))

        measure_duration = self.model.readout_time_us
        start_us = self._advance(targets, measure_duration)
        self.Nmeas += interior_count + boundary_count
        self.ops.append(ScheduleOp(
            kind="mid_circuit_measure", start_us=start_us, duration_us=measure_duration,
            targets=tuple(targets), detail={"num_ancilla": interior_count + boundary_count}
        ))

    def record_merge(self, placement: Placement, qubit_a: int, qubit_b: int, *, rounds: int | None = None):

        """
        Execute surgery merge: load bridge from reservior then run rounds
        """

        unit_um = self.model.gate_pair_dist_um
        pa = next(p for p in placement.patches if p.qubit == qubit_a)
        pb = next(p for p in placement.patches if p.qubit == qubit_b)
        merged_patch, anchor, origin_qubit = merge_patches(placement, qubit_a, qubit_b)

        before_positions = {
            pos for p in (pa, pb) for _, pos in p.patch.embed(p.anchor, unit_um)
        }
        after_positions = {pos for _, pos in merged_patch.embed(anchor, unit_um)}
        new_positions = after_positions - before_positions
        removed_positions = before_positions - after_positions

        removed_old_ids = [
            (p.qubit, atom.local)
            for p in (pa, pb)
            for atom, pos in p.patch.embed(p.anchor, unit_um)
            if pos in removed_positions
        ]

        if new_positions or removed_positions:
            handover = self.model.handover_time_us
            start_us = self._advance(removed_old_ids, 2 * handover)
            self.Nh += 2 * (len(new_positions) + len(removed_positions))
            self.ops.append(ScheduleOp(
                kind="merge_handover", start_us=start_us, duration_us=2 * handover,
                targets=tuple(removed_old_ids),
                detail={"qubits": [qubit_a, qubit_b], "new": len(new_positions), "removed": len(removed_positions),
                        "merged_patch": merged_patch, "anchor": anchor, "origin_qubit": origin_qubit}
            ))

        before_ids = {(qubit_a, a.local) for a in pa.patch.atoms} | {(qubit_b, b.local) for b in pb.patch.atoms}
        after_ids = {(origin_qubit, a.local) for a in merged_patch.atoms}
        self.retarget_atoms(remove=before_ids, add=after_ids)

        if rounds is None:
            rounds = merged_patch.distance
        targets = list(after_ids)
        for _ in range(rounds):
            self._record_syndrome_round(merged_patch, targets)

        return origin_qubit, merged_patch, anchor

    def record_split(self, merged_patch: Patch, merged_anchor: Position, origin_qubit: int, original_a: Patch, original_b: Patch) -> None:

        """
        Reverse surgery merge
        """

        unit_um = self.model.gate_pair_dist_um

        merged_positions = {pos for _, pos in merged_patch.embed(merged_anchor, unit_um)}
        original_positions = {
            pos for p in (original_a, original_b) for _, pos in p.patch.embed(p.anchor, unit_um)
        }
        new_positions = original_positions - merged_positions
        removed_positions = merged_positions - original_positions

        removed_old_ids = [
            (origin_qubit, atom.local)
            for atom, pos in merged_patch.embed(merged_anchor, unit_um)
            if pos in removed_positions
        ]

        if new_positions or removed_positions:
            handover = self.model.handover_time_us
            start_us = self._advance(removed_old_ids, 2 * handover)
            self.Nh += 2 * (len(new_positions) + len(removed_positions))
            self.ops.append(ScheduleOp(
                kind="split_handover", start_us=start_us, duration_us=2 * handover,
                targets=tuple(removed_old_ids),
                detail={"qubits": [original_a.qubit, original_b.qubit], "new": len(new_positions), "removed": len(removed_positions),
                        "origin_qubit": origin_qubit, "restored": [original_a, original_b]}
            ))

        merged_ids = {(origin_qubit, a.local) for a in merged_patch.atoms}
        restored_ids = (
            {(original_a.qubit, a.local) for a in original_a.patch.atoms} | 
            {(original_b.qubit, a.local) for a in original_b.patch.atoms}
        )
        self.retarget_atoms(remove=merged_ids, add=restored_ids)

def simulate_one_qubit_circuit(circuit: Circuit, placement: Placement, model: Model) -> Schedule:

    schedule = Schedule(model=model, all_atoms=tuple(_all_atom_ids(placement)))
    for stage_idx in range(len(circuit.stages)):
        schedule.record_1q_stage(circuit, stage_idx, placement)

    return schedule

def circuit_fidelity(schedule: Schedule) -> float:

    """
    Estimate circuit fidelity using 
    F = F1^Ng1 * Fg^Nsp * F2^Ng2 * Fh^Nh * prod_q exp(-t_idle_q / T2*)
    Crosstalk and transport heating not modelled
    """

    model = schedule.model
    f = model.single_qubit_fidelity ** schedule.Ng1
    f *= model.carrier_fidelity ** schedule.Nsp
    f *= model.cz_fidelity ** schedule.Ng2
    f *= model.handover_fidelity ** schedule.Nh
    f *= model.readout_fidelity ** schedule.Nmeas

    t2_us = model.t2_s * 1e6
    for t_idle in schedule.idle_us.values():
        f *= math.exp(-t_idle / t2_us)

    return f

_CAND_EDGES = (Edge.RIGHT, Edge.LEFT, Edge.TOP, Edge.BOTTOM)

_CONFLICT_COST = {
    "crossing": 3.0, # cannot park
    "shared_target": 2.0, # cannot park
    "shared_source": 0.5 # can park
}

def _cand_anchor(pa: PlacedPatch, edge: Edge, width: float, height: float) -> Position:

    if edge is Edge.RIGHT:
        return Position(pa.anchor.x + width, pa.anchor.y, pa.anchor.z)
    if edge is Edge.LEFT:
        return Position(pa.anchor.x - width, pa.anchor.y, pa.anchor.z)
    if edge is Edge.TOP:
        return Position(pa.anchor.x, pa.anchor.y + height, pa.anchor.z)
    return Position(pa.anchor.x, pa.anchor.y - height, pa.anchor.z)

def _conflict_cost(move: Move, committed: list[Move]) -> float:
    return sum(
        _CONFLICT_COST.get(reason[1], 0.0)
        for other in committed
        if (reason := conflict_reason(move, other)) is not None
    )

def _site_is_free(target_anchor: Position, radius: float, placement: Placement, exclude_qubit: int, *, eps: float = 1e-6) -> bool:

    for p in placement.patches:
        if p.qubit == exclude_qubit: continue

        other_radius = p.patch.radius_um(placement.model.gate_pair_dist_um)
        d = math.hypot(target_anchor.x - p.anchor.x, target_anchor.y - p.anchor.y)
        if d < radius + other_radius - eps: return False

    return True

def _retarget_adjacent(placement: Placement, qa: int, qb: int, committed: list[Move], *, lambda_par: float = 25.0) -> Placement:

    """
    Move qb to sit to the right of qa
    """

    pa = next(p for p in placement.patches if p.qubit == qa)
    pb = next(p for p in placement.patches if p.qubit == qb)
    if pa.patch.distance != pb.patch.distance:
        raise ValueError(
            "Patches have different distance"
        )

    unit_um = placement.model.gate_pair_dist_um
    min_x, min_y, max_x, max_y = pa.patch.local_bounds
    width = (max_x - min_x) * unit_um
    height = (max_y - min_y) * unit_um
    radius = pb.patch.radius_um(unit_um)

    best_move, best_score = None, math.inf
    for edge in _CAND_EDGES:
        target_anchor = _cand_anchor(pa, edge, width, height)
        if not _site_is_free(target_anchor, radius, placement, exclude_qubit=qb): continue

        candidate = Move(
            qubit=qb, source=pb.anchor, target=target_anchor,
            clearance_um=pb.patch.radius_um(unit_um)
        )

        score = lambda_par * _conflict_cost(candidate, committed) + candidate.distance_um()
        if score < best_score:
            best_score, best_move = score, candidate

    patches = tuple(
        PlacedPatch(qubit=p.qubit, patch=p.patch, anchor=best_move.target if p.qubit == qb else p.anchor)
        for p in placement.patches
    )

    return Placement(model=placement.model, patches=patches), best_move

def simulate_circuit(circuit: Circuit, placement: Placement, model: Model) -> tuple[Schedule, Placement]:

    schedule = Schedule(model=model, all_atoms=set(_all_atom_ids(placement)))
    current = placement

    for stage_idx, stage in enumerate(circuit.stages):
        ops_before = len(schedule.ops)
        two_qubit_gates = [g for g in stage if g.type is GateType.TWO_QUBIT]

        if not two_qubit_gates:
            schedule.record_1q_stage(circuit, stage_idx, current)

        else:

            involved = [q for g in two_qubit_gates for q in g.qubits]
            if len(involved) != len(set(involved)):
                raise NotImplementedError("A qubit appears in >1 2Q gate")

            target_placement = current
            committed_moves: list[Move] = []
            for gate in two_qubit_gates:
                qa, qb = gate.qubits
                target_placement, move = _retarget_adjacent(target_placement, qa, qb, committed_moves)
                committed_moves.append(move)

            frames = legalise_frames(moves_between(current, target_placement))
            for frame in frames:
                schedule.record_move_frame(frame, current)
            current = target_placement

            for gate in two_qubit_gates:
                qa, qb = gate.qubits
                origin_qubit, merged_patch, anchor = schedule.record_merge(current, qa, qb)
                pa = next(p for p in current.patches if p.qubit == qa)
                pb = next(p for p in current.patches if p.qubit == qb)
                
                schedule.record_split(merged_patch, anchor, origin_qubit, pa, pb)

        for op in schedule.ops[ops_before:]:
            op.detail["stage"] = stage_idx

    return schedule, current