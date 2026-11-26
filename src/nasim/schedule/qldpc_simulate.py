from __future__ import annotations

from nasim.circuit import Circuit, GateType
from nasim.device import Device
from nasim.model import Model
from nasim.move import Move, moves_between
from nasim.placement import Placement
from nasim.resource import Factory
from nasim.schedule.core import Schedule, ScheduleOp
from nasim.schedule.ldst import LoadStoreAncilla
from nasim.schedule.magic import _resolve_magic_requests
from nasim.schedule.qldpc_interface import ComputeResidency
from nasim.schedule.qldpc_memory import MemoryBlock, MemoryOccupancy
from nasim.schedule.qldpc_patches import (
    _all_potential_atom_ids,
    ensure_loaded_as_patches,
    manage_qldpc_patches
)
from nasim.schedule.qldpc_placement import build_initial_memory_state
from nasim.schedule.routing import _legalise_by_zone, _retarget_pair_into_entanglement_zone

def _run_in_waves(current, device, schedule, circuit, stage_idx, blocks, occupancies, ancillas, residency, *,
                  qldpc_code_distance, surface_distance, compute_capacity, units, execute):

    remaining = list(units)
    while remaining:
        current = manage_qldpc_patches(current, device, schedule, circuit, stage_idx, blocks, occupancies, ancillas, residency,
                                       {q for u in remaining for q in u}, qldpc_code_distance=qldpc_code_distance,
                                       surface_distance=surface_distance, compute_capacity=compute_capacity)

        batch, batch_qubits = [], set()
        for u in remaining:
            if len(batch_qubits | set(u)) > compute_capacity:
                if not batch:
                    raise ValueError("too small for single op")
                break

            batch.append(u)
            batch_qubits |= set(u)

        for u in batch:
            remaining.remove(u)

        current = ensure_loaded_as_patches(current, device, schedule, sorted(batch_qubits), circuit, stage_idx, blocks,
                                           occupancies, ancillas, residency, qldpc_code_distance=qldpc_code_distance,
                                           surface_distance=surface_distance, compute_capacity=compute_capacity)
        current = execute(current, batch)

    return current

def simulate_qldpc_circuit(
        circuit: Circuit, model: Model, device: Device, blocks: dict[int, MemoryBlock],
        factories: list[Factory] | None = None,
        *, qldpc_code_distance: int, surface_distance: int, compute_capacity: int, track_idle: bool = True
) -> tuple[Schedule, Placement, dict[int, int], dict[int, MemoryOccupancy], ComputeResidency]:

    assignment, occupancies = build_initial_memory_state(circuit, blocks)
    ancillas = {b: LoadStoreAncilla(memory_block_id=b) for b in blocks}
    residency = ComputeResidency()

    all_atoms = _all_potential_atom_ids(circuit.num_qubits, blocks, surface_distance)
    schedule = Schedule(model=model, all_atoms=all_atoms, device=device, track_idle=track_idle)
    current = Placement(model=model, patches=())

    for stage_idx, stage in enumerate(circuit.stages):
        ops_before = len(schedule.ops)
        two_qubit_gates = [g for g in stage if g.type is GateType.TWO_QUBIT]
        magic_gates = [g for g in stage if g.type is GateType.MAGIC_CONSUME]

        if magic_gates:
            if len(magic_gates) != len(stage):
                raise NotImplementedError("Mixes magic with other types")
            if not factories:
                raise ValueError("Requires factories")

            def _execute_magic(current, batch):
                qs = sorted({q for u in batch for q in u})
                return _resolve_magic_requests(current, device, schedule, factories, qs)

            units = [(g.qubits[0],) for g in magic_gates]
            current = _run_in_waves(current, device, schedule, circuit, stage_idx, blocks, occupancies,
                                    ancillas, residency, qldpc_code_distance=qldpc_code_distance, surface_distance=surface_distance,
                                    compute_capacity=compute_capacity, units=units, execute=_execute_magic)

        elif not two_qubit_gates:
            one_qubit_targets = sorted({g.qubits[0] for g in stage if g.type is GateType.ONE_QUBIT})
            if one_qubit_targets:
                duration = model.single_qubit_duration_us
                start_us = schedule._advance([], duration)
                schedule.ops.append(ScheduleOp(
                    kind="1q_stage", start_us=start_us, duration_us=duration,
                    targets=(), detail={"qubits": one_qubit_targets}
                ))

        else:
            involved_list = [q for g in two_qubit_gates for q in g.qubits]
            if len(involved_list) != len(set(involved_list)):
                raise NotImplementedError("A qubit appears in >1 2Q gate")
            involved = set(involved_list)

            def execute_pairs(current, batch_pairs):

                target_placement = current
                committed_moves: list[Move] = []
                for qa, qb in batch_pairs:
                    target_placement, moves = _retarget_pair_into_entanglement_zone(
                        target_placement, device, qa, qb, committed_moves
                    )
                    committed_moves.extend(moves)

                moves = moves_between(current, target_placement)
                moving_qubits = {m.qubit for m in moves}
                stat = [p.anchor for p in current.patches if p.qubit not in moving_qubits]
                rounds = _legalise_by_zone(moves, stat, device)
                for lanes in rounds:
                    schedule.record_move_round(lanes, current)
                current = target_placement

                merge_results = schedule.record_merge_batch(current, batch_pairs)

                splits = []
                for (qa,qb), (origin_qubit, merged_patch, anchor) in zip(batch_pairs, merge_results):
                    pa = next(p for p in current.patches if p.qubit == qa)
                    pb = next(p for p in current.patches if p.qubit == qb)
                    splits.append((merged_patch, anchor, origin_qubit, pa, pb))
                schedule.record_split_batch(splits)

                return current

            units = [tuple(g.qubits) for g in two_qubit_gates]
            current = _run_in_waves(current, device, schedule, circuit, stage_idx, blocks, occupancies,
                                    ancillas, residency, qldpc_code_distance=qldpc_code_distance, surface_distance=surface_distance,
                                    compute_capacity=compute_capacity, units=units, execute=execute_pairs)

        for op in schedule.ops[ops_before:]:
            op.detail["stage"] = stage_idx

    schedule.finalize_idle()
    return schedule, current, assignment, occupancies, residency