from __future__ import annotations

from dataclasses import dataclass, field

from nasim.circuit import Circuit
from nasim.schedule.core import Schedule
from nasim.schedule.ldst import LoadStoreAncilla, record_load, record_store
from nasim.schedule.qldpc_memory import MemoryBlock, MemoryOccupancy

@dataclass
class ComputeResidency:

    """
    Which (memory_block_id, slot) a program qubit currently resident in
    compute should be stored back into.
    """

    home: dict[int, tuple[int, int]] = field(default_factory=dict)

    def mark_loaded(self, program_qubit: int, *, block_id: int, slot: int):
        self.home[program_qubit] = (block_id, slot)

    def mark_stored(self, program_qubit: int):
        self.home.pop(program_qubit, None)

    def is_resident(self, program_qubit: int) -> bool:
        return program_qubit in self.home

    def resident_qubits(self) -> list[int]:
        return list(self.home.keys())

    def home_of(self, program_qubit: int) -> tuple[int, int]:
        return self.home[program_qubit]

def _find_block_holding(program_qubit: int, occupancies: dict[int, MemoryOccupancy]) -> int | None:

    for block_id, occ in occupancies.items():
        if occ.slot_of(program_qubit) is not None: return block_id

    return None

def _stages_until_next_use(circuit: Circuit, qubit: int, from_stage_idx: int) -> int | None:

    for ofs, stage in enumerate(circuit.stages[from_stage_idx + 1:], start=1):
        for gate in stage:
            if qubit in gate.qubits: return ofs

def ensure_loaded(schedule: Schedule, qubits: list[int], blocks: dict[int, MemoryBlock],
                  occupancies: dict[int, MemoryOccupancy], ancillas: dict[int, LoadStoreAncilla],
                  residency: ComputeResidency, *, code_distance: int):

    """
    Make sure gate's qubits are in compute, loading any still in mem
    """

    for q in qubits:
        if residency.is_resident(q): continue

        block_id = _find_block_holding(q, occupancies)
        if block_id is None: continue

        block = blocks[block_id]
        occupancy = occupancies[block_id]
        ancilla = ancillas[block_id]
        slot = occupancy.slot_of(q)

        record_load(schedule, block, occupancy, ancilla, slot=slot, code_distance=code_distance)
        residency.mark_loaded(q, block_id=block_id, slot=slot)

def manage_qldpc_occupancy(schedule: Schedule, circuit: Circuit, stage_idx: int,
                           blocks: dict[int, MemoryBlock], occupancies: dict[int, MemoryOccupancy],
                           ancillas: dict[int, LoadStoreAncilla], residency: ComputeResidency,
                           involved: set[int], *, code_distance: int, compute_capacity: int):

    """
    qSIEVE's greedy load/store policy. Compute qubit stored if has no future use or compute is at
    capacity and a stored qubit is used sooner
    Then load earliest stored qubit into any free slot (ahead of time)
    """

    for block_id, block in blocks.items():
        occupancy = occupancies[block_id]
        ancilla = ancillas[block_id]
        if not ancilla.is_free(schedule.clock_us): continue

        def next_use(q: int) -> int | None:
            return _stages_until_next_use(circuit, q, stage_idx)

        resident_here = [q for q in residency.resident_qubits() if residency.home_of(q)[0] == block_id]
        waiting_here = [q for q in occupancy.slots.values() if q is not None]
        evictable = [q for q in resident_here if q not in involved]

        to_store = None
        dead = [q for q in evictable if next_use(q) is None]
        if dead:
            to_store = dead[0]
        elif len(residency.resident_qubits()) >= compute_capacity and waiting_here and evictable:

            urgent_cands = [q for q in waiting_here if next_use(q) is not None]
            most_urgent_waiting = min(urgent_cands, key=next_use, default=None)
            if most_urgent_waiting is not None:
                worst_resident = max(evictable, key=lambda q: (next_use(q) is None, next_use(q) or 0))
                if next_use(worst_resident) is None or next_use(most_urgent_waiting) < next_use(worst_resident):
                    to_store = worst_resident

        if to_store is not None:
            _, slot = residency.home_of(to_store)
            record_store(schedule, block, occupancy, ancilla, slot=slot, program_qubit=to_store,
                         code_distance=code_distance)
            residency.mark_stored(to_store)
            waiting_here.append(to_store)

        if ancilla.is_free(schedule.clock_us) and len(residency.resident_qubits()) < compute_capacity:
            cands = [q for q in waiting_here if next_use(q) is not None]
            if cands:
                cand = min(cands, key=next_use)
                slot = occupancy.slot_of(cand)
                record_load(schedule, block, occupancy, ancilla, slot=slot, code_distance=code_distance)
                residency.mark_loaded(cand, block_id=block_id, slot=slot)