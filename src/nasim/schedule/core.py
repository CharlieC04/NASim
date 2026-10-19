from __future__ import annotations

import math
from dataclasses import dataclass, field

from nasim.circuit import Circuit, GateType
from nasim.model import Model
from nasim.move import Move, move_duration_us
from nasim.placement import Placement
from nasim.geometry import Position
from nasim.surface_code import AtomRole, Patch, _ancilla_weight_counts
from nasim.lattice_surgery import merge_patches
from nasim.device import Device

AtomId = tuple[int, tuple[int, int]]

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
    Accumulate time sequence of operations:
    - Ng1: physical 1q gate count (1 per stage)
    - Nsp: carrier-background exposures
    - Ng2: physical 2q gate count
    - Nh: handover count
    - idle_us: cum idle time per atom
    """

    model: Model
    all_atoms: set[AtomId]
    device: Device | None = None
    clock_us: float = 0.0
    ops: list[ScheduleOp] = field(default_factory=list)

    Ng1: int = 0
    Nsp: int = 0
    Ng2: int = 0
    Nh: int = 0
    Nmeas: int = 0
    Nxtalk: int = 0

    idle_us: dict[AtomId, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.all_atoms = set(self.all_atoms)
        for atom in self.all_atoms:
            self.idle_us.setdefault(atom, 0.0)

    def _entanglement_zone_bystanders(self, placement: Placement, exclude: set[int]) -> tuple[AtomId, ...]:

        if self.device is None or not self.device.zones: return ()

        ent_zones = [z for z in self.device.zones if z.kind == "entanglement"]
        if not ent_zones: return ()

        bystanders: list[AtomId] = []
        for p in placement.patches:
            if p.qubit in exclude: continue
            if any(z.contains(p.anchor) for z in ent_zones):
                bystanders.extend((p.qubit, atom.local) for atom in p.patch.atoms)

        return tuple(bystanders)

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
        Execute 1Q stage as shared global pulse
        """

        stage = circuit.stages[stage_idx]
        if any(g.type is not GateType.ONE_QUBIT for g in stage):
            raise NotImplementedError("Stage contains unsupported type")

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
            targets=tuple(targets), detail={"stage": stage_idx, "qubits": [g.qubits[0] for g in stage]}
        ))

    def record_move_frame(self, frame: list[Move], placement_before: Placement) -> None:

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
            targets=tuple(targets), detail={"qubits": [m.qubit for m in frame], "moves": tuple(frame)}
        ))

    def _record_syndrome_round(self, merged_patch: Patch, targets: list[AtomId], bystanders: tuple[AtomId, ...] = ()):

        """
        4 seq CZ pulses
        """ 

        interior_count, boundary_count = _ancilla_weight_counts(merged_patch)
        cz_duration = self.model.cz_duration_us

        for step in range(4):
            start_us = self._advance(targets, cz_duration)
            active = interior_count + (boundary_count if step < 2 else 0)
            self.Ng2 += active
            self.Nxtalk += len(bystanders)
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
        Perform surgery merge, asm bridge loaded from reserviour
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
        bystanders = self._entanglement_zone_bystanders(placement, exclude={qubit_a, qubit_b})
        for _ in range(rounds):
            self._record_syndrome_round(merged_patch, targets, bystanders)

        return origin_qubit, merged_patch, anchor

    def record_split(self, merged_patch: Patch, merged_anchor: Position, origin_qubit: int,
                     original_a: Patch, original_b: Patch):

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
                detail={"qubits": [original_a.qubit, original_b.qubit], "new": len(new_positions), 
                        "removed": len(removed_positions),
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

def circuit_fidelity(schedule: Schedule, *, d: int) -> float:

    """
    WIP
    """

    model = schedule.model
    f = model.single_qubit_fidelity ** schedule.Ng1
    f *= model.carrier_fidelity ** schedule.Nsp
    f *= model.handover_fidelity ** schedule.Nh
    f *= model.carrier_fidelity ** schedule.Nxtalk

    t2_us = model.t2_s * 1e6
    for t_idle in schedule.idle_us.values():
        f *= math.exp(-t_idle / t2_us)

    num_rounds = sum(1 for op in schedule.ops if op.kind == "mid_circuit_measure")
    if num_rounds:
        d_e = (d + 1) // 2 if d % 2 == 1 else d // 2
        p = 1.0 - model.cz_fidelity
        p_th = 0.0057 # fowler paper
        pl_per_round = min(1.0, 0.03 * (p / p_th) ** d_e)
        f *= (1.0 - pl_per_round) ** num_rounds

    return f