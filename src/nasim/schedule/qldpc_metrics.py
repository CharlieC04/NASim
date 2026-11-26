from __future__ import annotations

from nasim.placement import Placement
from nasim.schedule.core import Schedule
from nasim.schedule.qldpc_memory import MemoryBlock
from nasim.surface_code import Patch

def qldpc_spacetime_volume_us(schedule: Schedule, blocks: dict[int, MemoryBlock], surface_dist: int) -> float:

    patch_atom_count = len(Patch.rotated(surface_dist).atoms)
    mem_atom_count = sum(len(block.atom_ids()) for block in blocks.values())
    total_us = schedule.clock_us

    events: list[tuple[float, int]] = []
    for op in schedule.ops:
        if op.kind == "ldst_load":
            events.append((op.start_us, 1))
        elif op.kind == "ldst_store":
            events.append((op.start_us + op.duration_us, -1))
    events.sort(key=lambda e: e[0])

    volume = mem_atom_count * total_us
    live_patches = 0
    last_t = 0.0
    for t, delta in events:
        volume += live_patches * patch_atom_count * (t-last_t)
        live_patches += delta
        last_t = t
    volume += live_patches * patch_atom_count * (total_us - last_t)

    return volume

def baseline_spacetime_volume_us(schedule: Schedule, placement: Placement, surface_dist: int) -> float:

    patch_atom_count = len(Patch.rotated(surface_dist).atoms)
    num_qubits = len(placement.patches)
    return num_qubits * patch_atom_count * schedule.clock_us