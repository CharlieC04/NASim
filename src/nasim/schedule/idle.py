from __future__ import annotations

import math

from nasim.circuit import Circuit, GateType
from nasim.device import Device
from nasim.model import Model
from nasim.move import Move, move_duration_us
from nasim.placement import Placement, PlacedPatch
from nasim.schedule.routing import _nearest_free_zone_site

def _stages_until_next_2q(circuit: Circuit, qubit: int, from_stage_idx: int) -> int | None:

    for ofs, stage in enumerate(circuit.stages[from_stage_idx + 1:], start=1):
        for gate in stage:
            if gate.type is GateType.TWO_QUBIT and qubit in gate.qubits:
                return ofs

def _idle_qubit_management(circuit: Circuit, stage_idx: int, placement: Placement, device: Device, 
                           model: Model, involved: set[int]):

    """
    ZAP: compare crosstalk from staying in Ez vs move cost
    """

    ent_zones = [z for z in device.zones if z.kind == "entanglement"]
    storage_zones = [z for z in device.zones if z.kind == "storage"]
    if not ent_zones or not storage_zones: return placement, []
    ent_zone, storage_zone = ent_zones[0], storage_zones[0]

    resident = [
        p.qubit for p in placement.patches
        if p.qubit >= 0 and p.qubit not in involved and ent_zone.contains(p.anchor)
    ]
    if not resident: return placement, []

    t2_us = model.t2_s * 1e6
    neg_ln_carrier = -math.log(model.carrier_fidelity)
    neg_ln_handover = -math.log(model.handover_fidelity)

    moves: list[Move] = []
    working = placement
    for q in resident:
        p = next(pp for pp in working.patches if pp.qubit == q)
        radius = p.patch.radius_um(model.gate_pair_dist_um)
        target = _nearest_free_zone_site(storage_zone, working, p.patch, model.gate_pair_dist_um, p.anchor, 
                                         radius=radius, exclude_qubit=q)

        if target is None: continue

        k = _stages_until_next_2q(circuit, q, stage_idx)
        candidate = Move(qubit=q, source=p.anchor, target=target, clearance_um=radius)
        move_time = move_duration_us(candidate, model)
        n_tr = 4 if k is not None else 2
        cost_leave = n_tr * neg_ln_handover + (n_tr / 2) * (move_time / t2_us)
        cost_stay = (k * neg_ln_carrier) if k is not None else math.inf
        should_return = cost_stay > cost_leave

        if should_return:
            patches = tuple(
                PlacedPatch(qubit=pp.qubit, patch=pp.patch, anchor=target if pp.qubit == q else pp.anchor)
                for pp in working.patches
            )
            working = Placement(model=working.model, patches=patches)
            moves.append(candidate)

    return working, moves