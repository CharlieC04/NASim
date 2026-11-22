from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from nasim.geometry import Position
from nasim.qldpc import GenBicycleCode
from nasim.schedule.core import AtomId, Schedule, ScheduleOp
from nasim.schedule.systolic import QSIEVE_SPACING_UM, route_stops

class QldpcRole(Enum):

    X_CHECK = 0
    Z_CHECK = 1
    A_DATA = 2
    B_DATA = 3

def memory_atom_id(block_id: int, role: QldpcRole, index: int, *, l: int, m: int) -> AtomId:

    row = index % m
    col = role.value * l + index // m
    return (block_id, (row, col))

@dataclass(frozen=True)
class MemoryBlock:

    """
    One instance of qLDPC code in mem zone
    """

    block_id: int
    code: GenBicycleCode
    anchor: Position

    def atom_ids(self) -> list[AtomId]:
        ids: list[AtomId] = []
        for role in QldpcRole:
            for i in range(self.code.lm):
                ids.append(memory_atom_id(self.block_id, role, i, l=self.code.l, m=self.code.m))
        return ids

@dataclass
class MemoryOccupancy:

    """
    Which program qubit occupies a block's logical slots
    """

    block_id: int
    slots: dict[int, int | None]

    @classmethod
    def empty(cls, block: MemoryBlock) -> MemoryOccupancy:
        return cls(block_id=block.block_id, slots={i: None for i in range(block.code.k)})

    def store(self, slot: int, program_qubit: int):

        if slot not in self.slots: raise ValueError("slot does not exist")
        if self.slots[slot] is not None: raise ValueError("slot occupied")

        self.slots[slot] = program_qubit

    def evict(self, slot: int) -> int:
        occupant = self.slots.get(slot)
        if occupant is None: raise ValueError("slot empty")

        self.slots[slot] = None
        return occupant

    def free_slots(self) -> list[int]:
        return [s for s, q in self.slots.items() if q is None]

    def slot_of(self, program_qubit: int) -> int | None:
        for s, q in self.slots.items():
            if q == program_qubit: return s

        return None

def round_duration_us(code: GenBicycleCode, *, accel: float, spacing_um: float = QSIEVE_SPACING_UM) -> float:

    """
    Total time for one round of stab checks (X then Z acc to QS)
    """

    x_ofs = code.systolic_offsets(hx=True)
    z_ofs = code.systolic_offsets(hx=False)
    _, x_time = route_stops(x_ofs, spacing_um=spacing_um, accel=accel)
    _, z_time = route_stops(z_ofs, spacing_um=spacing_um, accel=accel)
    return x_time + z_time

def record_check_round_batch(schedule: Schedule, blocks: list[MemoryBlock], *, accel: float) -> None:

    """
    Run one round of checks across all mem blocks at once. Asm have same structure and share AOD schedule
    """

    if not blocks: return

    code = blocks[0].code
    if any(b.code != code for b in blocks):
        raise ValueError(
            "record_check_round_batch requires every memory block to share "
            "the same code (qSIEVE's shared-AOD-control assumption, 7.1)"
        )
    
    duration_us = round_duration_us(code, accel=accel)
    targets: list[AtomId] = [aid for block in blocks for aid in block.atom_ids()]
    start_us = schedule._advance(targets, duration_us)
    schedule.ops.append(ScheduleOp(
        kind="qldpc_check_round", start_us=start_us, duration_us=duration_us,
        targets=tuple(targets),
        detail={"block_ids": [b.block_id for b in blocks], "code": code.name}
    ))