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
from nasim.schedule.qldpc_interface import ComputeResidency, _find_block_holding
from nasim.schedule.qldpc_memory import MemoryBlock, MemoryOccupancy
from nasim.schedule.qldpc_patches import (
    _all_potential_atom_ids,
    assign_ancilla_ports,
    ensure_loaded_as_patches,
    manage_qldpc_patches
)
from nasim.schedule.qldpc_placement import build_initial_memory_state
from nasim.schedule.routing import _legalise_by_zone, _retarget_pair_into_entanglement_zone, _site_anchor
from nasim.surface_code import Patch

def _run_in_waves(current, device, schedule, circuit, stage_idx, blocks, occupancies, ancillas, residency, ports, *,
                  qldpc_code_distance, surface_distance, compute_capacity, units, execute):

    remaining = list(units)
    while remaining:
        current = manage_qldpc_patches(current, device, schedule, circuit, stage_idx, blocks, occupancies, ancillas, residency,
                                       {q for u in remaining for q in u}, ports, qldpc_code_distance=qldpc_code_distance,
                                       surface_distance=surface_distance, compute_capacity=compute_capacity)

        # Global compute_capacity isn't the only cap: each block's ld/st
        # ancilla only has len(ports[block_id]) fixed ports, so a batch
        # can be globally within capacity yet still need more
        # concurrently-resident qubits from one particular block than
        # that block has ports for (e.g. several gates all touching the
        # same block). What matters is how many of the BATCH'S OWN qubits
        # are affiliated with each block -- whether still in memory
        # (occupancies) or already resident in compute (residency), since
        # either way they tie up one of that block's ports for the
        # batch's duration. Anyone NOT in this batch currently resident
        # there isn't a real obstacle, since ensure_loaded_as_patches can
        # always evict a non-batch resident one at a time to make room.
        # Track per-block batch-qubit counts as the batch grows, and stop
        # growing it -- deferring the rest to a later wave -- before any
        # single block's batch-qubit count would exceed its ports.
        def _home_block(q):
            if residency.is_resident(q):
                return residency.home_of(q)[0]
            return _find_block_holding(q, occupancies)

        batch, batch_qubits = [], set()
        batch_counts_by_block: dict[int, int] = {}
        for u in remaining:
            if len(batch_qubits | set(u)) > compute_capacity:
                if not batch:
                    raise ValueError("too small for single op")
                break

            candidate_counts = dict(batch_counts_by_block)
            fits = True
            for q in u:
                if q in batch_qubits:
                    continue
                block_id = _home_block(q)
                if block_id is None:
                    continue
                candidate_counts[block_id] = candidate_counts.get(block_id, 0) + 1
                if candidate_counts[block_id] > len(ports[block_id]):
                    fits = False
                    break

            if not fits:
                if not batch:
                    raise ValueError("too small for single op (exceeds a block's ancilla ports)")
                break

            batch_counts_by_block = candidate_counts
            batch.append(u)
            batch_qubits |= set(u)

        for u in batch:
            remaining.remove(u)

        current = ensure_loaded_as_patches(current, device, schedule, sorted(batch_qubits), circuit, stage_idx, blocks,
                                           occupancies, ancillas, residency, ports, qldpc_code_distance=qldpc_code_distance,
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
    ports = assign_ancilla_ports(device, blocks)

    unit_um = model.gate_pair_dist_um
    patch = Patch.rotated(surface_distance)
    reserved_sites = frozenset(_site_anchor(site, patch, unit_um) for sites in ports.values() for site in sites)

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
                return _resolve_magic_requests(current, device, schedule, factories, qs,
                                               reserved_sites=reserved_sites)

            units = [(g.qubits[0],) for g in magic_gates]
            current = _run_in_waves(current, device, schedule, circuit, stage_idx, blocks, occupancies,
                                    ancillas, residency, ports, qldpc_code_distance=qldpc_code_distance, 
                                    surface_distance=surface_distance,
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
                        target_placement, device, qa, qb, committed_moves, reserved_sites=reserved_sites
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
                                    ancillas, residency, ports,
                                      qldpc_code_distance=qldpc_code_distance, surface_distance=surface_distance,
                                    compute_capacity=compute_capacity, units=units, execute=execute_pairs)

        for op in schedule.ops[ops_before:]:
            op.detail["stage"] = stage_idx

    schedule.finalize_idle()
    return schedule, current, assignment, occupancies, residency