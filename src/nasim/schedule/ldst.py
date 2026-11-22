from __future__ import annotations

from dataclasses import dataclass

from nasim.schedule.core import AtomId, Schedule, ScheduleOp
from nasim.schedule.qldpc_memory import MemoryBlock, MemoryOccupancy

# using qSIEVE table 4
LOAD_LOGICAL_CYCLES = 2
STORE_LOGICAL_CYCLES = 1

def load_duration_us(qec_round_duration_us: float, code_distance: int) -> float:
    return LOAD_LOGICAL_CYCLES * code_distance * qec_round_duration_us

def store_duration_us(qec_round_duration_us: float, code_distance: int) -> float:
    return STORE_LOGICAL_CYCLES * code_distance * qec_round_duration_us

@dataclass
class LoadStoreAncilla:

    """
    LD/ST interface between one qLDPC memory block and surface code compute region.
    Asm memory block can serve one load or store at a time
    """

    memory_block_id: int
    busy_until_us: float = 0.0
    in_flight_qubit: int | None = None

    def is_free(self, now_us: float) -> bool:
        return now_us >= self.busy_until_us

    def begin(self, *, qubit: int, now_us: float, duration_us: float) -> None:

        if not self.is_free(now_us):
            raise ValueError('finish called before op done')

        self.in_flight_qubit = qubit
        self.busy_until_us = now_us + duration_us

    def finish(self, now_us: float):
        if now_us < self.busy_until_us:
            raise ValueError('finish called before op done')

        self.in_flight_qubit = None

def _check_ancilla_matches(ancilla: LoadStoreAncilla, block: MemoryBlock) -> None:
    if ancilla.memory_block_id != block.block_id:
        raise ValueError("block id no match")

def record_load(schedule: Schedule, block: MemoryBlock, occupancy: MemoryOccupancy, 
                ancilla: LoadStoreAncilla, *, slot: int, code_distance: int) -> int:

    """
    Move program qubit in slot out of mem and into compute. Have one-aat-block contraint
    """

    _check_ancilla_matches(ancilla, block)
    if not ancilla.is_free(schedule.clock_us):
        raise ValueError("LD/ST ancilla busy")

    program_qubit = occupancy.slots.get(slot)
    if program_qubit is None:
        raise ValueError("slot empty")

    duration_us = load_duration_us(schedule.model.qec_round_duration_us, code_distance)
    targets: list[AtomId] = block.atom_ids()
    start_us = schedule._advance(targets, duration_us)

    ancilla.begin(qubit=program_qubit, now_us=start_us, duration_us=duration_us)
    ancilla.finish(schedule.clock_us)
    occupancy.evict(slot)

    schedule.ops.append(ScheduleOp(
        kind="ldst_load", start_us=start_us, duration_us=duration_us,
        targets=tuple(targets), detail={"block_id": block.block_id, "slot": slot, "qubit": program_qubit}
    ))

    return program_qubit

def record_store(schedule: Schedule, block: MemoryBlock, occupancy: MemoryOccupancy,
                 ancilla: LoadStoreAncilla, *, slot: int, program_qubit: int, code_distance: int) -> None:

    """
    Move qubit from compute to memory
    """

    _check_ancilla_matches(ancilla, block)
    if not ancilla.is_free(schedule.clock_us):
        raise ValueError("LD/ST ancilla busy")

    duration_us = store_duration_us(schedule.model.qec_round_duration_us, code_distance)
    targets: list[AtomId] = block.atom_ids()
    start_us = schedule._advance(targets, duration_us)

    ancilla.begin(qubit=program_qubit, now_us=start_us, duration_us=duration_us)
    ancilla.finish(schedule.clock_us)
    occupancy.store(slot, program_qubit)

    schedule.ops.append(ScheduleOp(
        kind="ldst_store", start_us=start_us, duration_us=duration_us,
        targets=tuple(targets), detail={"block_id": block.block_id, "slot": slot, "qubit": program_qubit}
    ))

def record_ldst_wave(schedule: Schedule, block_seqs: list[list[tuple]], *, code_dist: int):

    """
    Dispatch LD/ST seq in waves, since share AOD across blocks
    """

    block_seqs = [seq for seq in block_seqs if seq]
    if not block_seqs: return

    def op_dur(kind: str) -> float:
        fn = load_duration_us if kind == "load" else store_duration_us
        return fn(schedule.model.qec_round_duration_us, code_dist)

    wave_start = schedule.clock_us
    all_tgts: list[AtomId] = []
    dispatch: list[tuple[float, float, str, MemoryBlock, MemoryOccupancy, LoadStoreAncilla, int, int]] = []
    max_end = wave_start

    for seq in block_seqs:
        t = wave_start
        for kind, block, occupancy, ancilla, slot, qubit in seq:
            _check_ancilla_matches(ancilla, block)
            if not ancilla.is_free(t): raise ValueError("ancilla busy")

            d = op_dur(kind)
            dispatch.append((t, d, kind, block, occupancy, ancilla, slot, qubit))
            all_tgts.extend(block.atom_ids())
            t += d
        max_end = max(max_end, t)

    schedule._advance(all_tgts, max_end - wave_start)

    for start_us, d, kind, block, occupancy, ancilla, slot, qubit in dispatch:
        ancilla.begin(qubit=qubit, now_us=start_us, duration_us=d)
        ancilla.finish(start_us+d)
        if kind == "load": occupancy.evict(slot)
        else: occupancy.store(slot, qubit)

        schedule.ops.append(ScheduleOp(
            kind=f"ldst_{kind}", start_us=start_us, duration_us=d, targets=tuple(block.atom_ids()),
            detail={"block_id": block.block_id, "slot": slot, "qubit": qubit}
        ))