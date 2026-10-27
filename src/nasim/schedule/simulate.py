from __future__ import annotations

from nasim.circuit import Circuit, GateType
from nasim.device import Device
from nasim.model import Model
from nasim.move import Move, legalise_frames, moves_between
from nasim.placement import Placement
from nasim.resource import Factory
from nasim.schedule.core import Schedule, _all_atom_ids
from nasim.schedule.idle import _idle_qubit_management
from nasim.schedule.magic import _resolve_magic_requests
from nasim.schedule.routing import _retarget_adjacent, _retarget_pair_into_entanglement_zone, _legalise_by_zone

def simulate_circuit(circuit: Circuit, placement: Placement, model: Model, device: Device | None = None,
                     factories: list[Factory] | None = None) -> tuple[Schedule, Placement]:


    schedule = Schedule(model=model, all_atoms=set(_all_atom_ids(placement)), device=device)
    current = placement

    for stage_idx, stage in enumerate(circuit.stages):
        ops_before = len(schedule.ops)
        two_qubit_gates = [g for g in stage if g.type is GateType.TWO_QUBIT]
        magic_gates = [g for g in stage if g.type is GateType.MAGIC_CONSUME]

        if magic_gates:
            if len(magic_gates) != len(stage):
                raise NotImplementedError("Mixes magic with other types")

            if device is None or not factories:
                raise ValueError("Requires a device with entanglement zone and factories")

            qubits = [g.qubits[0] for g in magic_gates]
            current = _resolve_magic_requests(current, device, schedule, factories, qubits)

        elif not two_qubit_gates:
            schedule.record_1q_stage(circuit, stage_idx, current)

        else:

            involved = [q for g in two_qubit_gates for q in g.qubits]
            if len(involved) != len(set(involved)):
                raise NotImplementedError("A qubit appears in >1 2Q gate")

            stage_start = current
            working = current
            idle_moves: list[Move] = []
            if device is not None and device.zones:
                working, idle_moves = _idle_qubit_management(circuit, stage_idx, working, device, model, set(involved))

            target_placement = working
            committed_moves: list[Move] = list(idle_moves)
            for gate in two_qubit_gates:
                qa, qb = gate.qubits
                if device is not None and device.zones:
                    target_placement, moves = _retarget_pair_into_entanglement_zone(target_placement, device, 
                                                                                    qa, qb, committed_moves)
                    committed_moves.extend(moves)
                else:
                    target_placement, moves_result = _retarget_adjacent(target_placement, qa, qb, committed_moves)
                    committed_moves.extend(moves_result)

            moves = moves_between(current, target_placement)
            moving_qubits = {m.qubit for m in moves}
            stationary = [p.anchor for p in current.patches if p.qubit not in moving_qubits]
            rounds = _legalise_by_zone(moves, stationary, device)
            for lanes in rounds:
                schedule.record_move_round(lanes, current)
            current = target_placement

            pairs = [gate.qubits for gate in two_qubit_gates]
            merge_results = schedule.record_merge_batch(current, pairs)
            splits =[]
            for (qa, qb), (origin_qubit, merged_patch, anchor) in zip(pairs, merge_results):

                pa = next(p for p in current.patches if p.qubit == qa)
                pb = next(p for p in current.patches if p.qubit == qb)
                splits.append((merged_patch, anchor, origin_qubit, pa, pb))

            schedule.record_split_batch(splits)

        for op in schedule.ops[ops_before:]:
            op.detail["stage"] = stage_idx

    return schedule, current