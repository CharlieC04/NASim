from __future__ import annotations

import math

from nasim.device import Device
from nasim.geometry import Position
from nasim.placement import PlacedPatch, Placement, _vectors_incompat
from nasim.schedule.core import AtomId
from nasim.schedule.ldst import LoadStoreAncilla, record_load, record_store, record_ldst_wave
from nasim.schedule.qldpc_interface import (
    ComputeResidency, _find_block_holding, _stages_until_next_use
)
from nasim.schedule.qldpc_memory import MemoryBlock, MemoryOccupancy
from nasim.schedule.routing import _site_anchor, _site_is_free
from nasim.surface_code import Patch

def _all_potential_atom_ids(num_qubits: int, blocks: dict[int, MemoryBlock], surface_distance: int) -> set[AtomId]:

    atoms: set[AtomId] = {a for block in blocks.values() for a in block.atom_ids()}
    patch = Patch.rotated(surface_distance)
    for q in range(num_qubits):
        atoms |= {(q, a.local) for a in patch.atoms}
    return atoms

_nearest_entanglement_target_cache: dict[tuple, tuple[Position | None, float]] = {}

def _nearest_entanglement_target(ent_sites: tuple[Position, ...], pos: Position, patch: Patch, unit_um: float):

    key = (ent_sites, pos, patch.distance, unit_um)
    cached = _nearest_entanglement_target_cache.get(key)
    if cached is not None: return cached

    best_e, best_d = None, math.inf
    for e in ent_sites:
        anchor_e = _site_anchor(e, patch, unit_um)
        d = math.hypot(pos.x - anchor_e.x, pos.y - anchor_e.y)
        if d < best_d:
            best_d, best_e = d, anchor_e

    result = (best_e, best_d)
    _nearest_entanglement_target_cache[key] = result
    return result

def _free_site_for_new_patch(device: Device, placement: Placement, patch: Patch, unit_um: float, *,
                             lambda_par: float = 25.0) -> Position:

    """
    ZAP cost func applied to one qubit being loaded from mem
    """

    storage_zones = [z for z in device.zones if z.kind == "storage"]
    ent_zones = [z for z in device.zones if z.kind == "entanglement"]
    cand_zones = storage_zones if storage_zones else ent_zones
    ent_sites = tuple(s for z in ent_zones for s in z.sites)

    radius = patch.radius_um(unit_um)
    committed = []
    for p in placement.patches:
        p_rad = p.patch.radius_um(unit_um)
        tgt, _ = _nearest_entanglement_target(ent_sites, p.anchor, p.patch, unit_um)
        committed.append((p.anchor, tgt, p_rad))

    best_site, best_score, best_anchor = None, math.inf, None
    for zone in cand_zones:
        for site in zone.sites:
            cand = _site_anchor(site, patch, unit_um)
            if not _site_is_free(cand, radius, placement, exclude_qubit=-1): continue

            tgt, d_min = _nearest_entanglement_target(ent_sites, cand, patch, unit_um)
            conflicts = sum(1 for (src, dst, clr) in committed
                            if _vectors_incompat(cand, tgt, src, dst, radius, clr))
            score = lambda_par * conflicts + d_min
            if score < best_score:
                best_score, best_site, best_anchor = score, site, cand

    if best_anchor is None: raise ValueError("no free site")
    return best_anchor

def _materialise_patch(current: Placement, device: Device, program_qubit: int, *, surface_distance: int) -> Placement:

    patch = Patch.rotated(surface_distance)
    unit_um = current.model.gate_pair_dist_um
    anchor = _free_site_for_new_patch(device, current, patch, unit_um)
    new_patches = current.patches + (PlacedPatch(qubit=program_qubit, patch=patch, anchor=anchor),)

    return Placement(model=current.model, patches=new_patches)

def _dissolve_patch(current: Placement, program_qubit: int, ) -> Placement:

    remaining = tuple(p for p in current.patches if p.qubit != program_qubit)
    return Placement(model=current.model, patches=remaining)

def ensure_loaded_as_patches(current: Placement, device: Device, schedule, qubits: list[int],
                             circuit, stage_idx: int, blocks: dict[int, MemoryBlock],
                             occupancies: dict[int, MemoryOccupancy], ancillas: dict[int, LoadStoreAncilla],
                             residency: ComputeResidency, *, qldpc_code_distance: int, surface_distance: int,
                             compute_capacity: int) -> Placement:

    """
    Make sure each qubit has a patch, loading any still in mem
    """

    present = {p.qubit for p in current.patches}
    protected = set(qubits)

    per_block: dict[int, list[tuple]] = {}
    to_load: list[int] = []
    to_evict: list[int] = []

    for q in qubits:
        if q in present: continue

        block_id = _find_block_holding(q, occupancies)
        if block_id is None: continue

        if len(residency.resident_qubits()) >= compute_capacity:
            evictable = [r for r in residency.resident_qubits() if r not in protected]
            if not evictable: raise ValueError("at capacity and no evictables")

            worst = max(evictable, key=lambda r:(
                _stages_until_next_use(circuit, r, stage_idx) is None,
                _stages_until_next_use(circuit, r, stage_idx) or 0
            ))
            worst_block_id, worst_slot = residency.home_of(worst)
            worst_ancilla = ancillas[worst_block_id]
            if not worst_ancilla.is_free(schedule.clock_us):
                raise ValueError("worst ancilla busy")

            per_block.setdefault(worst_block_id, []).append(
                ("store", blocks[worst_block_id], occupancies[worst_block_id], worst_ancilla, worst_slot, worst)
            )
            residency.mark_stored(worst)
            to_evict.append(worst)

        slot = occupancies[block_id].slot_of(q)
        per_block.setdefault(block_id, []).append(
            ("load", blocks[block_id], occupancies[block_id], ancillas[block_id], slot, q)
        )
        residency.mark_loaded(q, block_id=block_id, slot=slot)
        to_load.append(q)
        present.add(q)

    record_ldst_wave(schedule, list(per_block.values()), code_dist=qldpc_code_distance)

    for q in to_evict:
        current = _dissolve_patch(current, q)
    for q in to_load:
        current = _materialise_patch(current, device, q, surface_distance=surface_distance)

    return current

def manage_qldpc_patches(current: Placement, device: Device, schedule, circuit, stage_idx: int,
                         blocks: dict[int, MemoryBlock], occupancies: dict[int, MemoryOccupancy],
                         ancillas: dict[int, LoadStoreAncilla], residency: ComputeResidency,
                         involved: set[int], *, qldpc_code_distance: int, surface_distance: int,
                         compute_capacity: int) -> Placement:

    """
    Use need driven load/store policy
    """

    per_block: dict[int, list[tuple]] = {}
    to_evict: list[int] = []
    to_load: list[int] = []

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

        block_seq: list[tuple] = []
        if to_store is not None:
            _, slot = residency.home_of(to_store)
            block_seq.append(("store", block, occupancy, ancilla, slot, to_store))
            residency.mark_stored(to_store)
            waiting_here.append(to_store)
            to_evict.append(to_store)


        pending_mandatory = len([q for q in involved if not residency.is_resident(q)])
        if (ancilla.is_free(schedule.clock_us) and len(residency.resident_qubits()) + pending_mandatory < compute_capacity):

            cands = [q for q in waiting_here if next_use(q) is not None]
            if cands:
                cand = min(cands, key=next_use)
                slot = occupancy.slot_of(cand)
                block_seq.append(("load", block, occupancy, ancilla, slot, cand))
                residency.mark_loaded(cand, block_id=block_id, slot=slot)
                to_load.append(cand)

        if block_seq:
            per_block[block_id] = block_seq

    record_ldst_wave(schedule, list(per_block.values()), code_dist=qldpc_code_distance)

    for q in to_evict:
        current = _dissolve_patch(current, q)
    for q in to_load:
        current = _materialise_patch(current, device, q, surface_distance=surface_distance)

    return current
