from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from nasim.device import Device
from nasim.move import Move, legalise_frames, moves_between
from nasim.placement import Placement, PlacedPatch
from nasim.resource import Factory
from nasim.schedule.core import Schedule, ScheduleOp, _data_atom_ids
from nasim.schedule.routing import _nearest_free_zone_site, _retarget_pair_into_entanglement_zone
from nasim.surface_code import Patch

def _ensure_placed(current: Placement, zone, resource_id: int, distance: int, reference_anchor) -> Placement:

    """
    Auto-place factory at nearest free zone site
    """

    if any(p.qubit == resource_id for p in current.patches): return current

    unit_um = current.model.gate_pair_dist_um
    patch = Patch.rotated(distance)
    radius = patch.radius_um(unit_um)

    site = _nearest_free_zone_site(zone, current, patch, unit_um, reference_anchor, radius=radius, exclude_qubit=resource_id)
    if site is None: raise ValueError("No free site to place factory")

    return Placement(model=current.model, patches=current.patches + (PlacedPatch(
        qubit=resource_id, patch=patch, anchor=site
    ),))

def _teleport_consume(current: Placement, device: Device, schedule: Schedule, qubit_a: int, qubit_b: int) -> Placement:

    """
    Route two qubits next to each other in Ez and consume via surgery
    """

    working, moves = _retarget_pair_into_entanglement_zone(current, device, qubit_a, qubit_b, [])
    if moves:
        realised = moves_between(current, working)
        moving_qubits = {m.qubit for m in realised}
        stationary = [p.anchor for p in current.patches if p.qubit not in moving_qubits]
        frames = legalise_frames(realised, stationary)
        for frame in frames:
            schedule.record_move_frame(frame, current)

    pa = next(p for p in working.patches if p.qubit == qubit_a)
    pb = next(p for p in working.patches if p.qubit == qubit_b)
    origin_qubit, merged_patch, anchor = schedule.record_merge(working, qubit_a, qubit_b)
    schedule.record_split(merged_patch, anchor, origin_qubit, pa, pb)

    return working

def _return_to_zone(current: Placement, schedule: Schedule, zone, resource_id: int) -> Placement:

    """
    Return factory to nearest free zone site after delivery
    """

    unit_um = schedule.model.gate_pair_dist_um
    p = next(pp for pp in current.patches if pp.qubit == resource_id)
    radius = p.patch.radius_um(unit_um)
    site = _nearest_free_zone_site(zone, current, p.patch, unit_um, p.anchor, radius=radius, exclude_qubit=resource_id)
    if site is None:
        raise ValueError("No free site for resource")

    if site == p.anchor: return current

    move = Move(qubit=resource_id, source=p.anchor, target=site, clearance_um=radius)
    stationary = [pp.anchor for pp in current.patches if pp.qubit != resource_id]
    frames = legalise_frames([move], stationary)
    for frame in frames:
        schedule.record_move_frame(frame, current)

    return Placement(
        model=current.model, patches=tuple(PlacedPatch(
            qubit=pp.qubit, patch=pp.patch, anchor=site if pp.qubit == resource_id else pp.anchor
        )
        for pp in current.patches
        )
    )

def _match_by_distance(current: Placement, a_ids: list[int], b_ids: list[int]) -> list[tuple[int, int]]:

    """
    Minimum-cost bipartite match. Used for check-stage redist and escape fact -> deliv
    """

    k = min(len(a_ids), len(b_ids))
    if k == 0: return []

    def pos(resource_id: int):
        return next(p.anchor for p in current.patches if p.qubit == resource_id)

    cost = np.array([[math.hypot(pos(a).x - pos(b).x, pos(a).y - pos(b).y) for b in b_ids] for a in a_ids])
    row_ind, col_ind = linear_sum_assignment(cost)
    return [(a_ids[i], b_ids[j]) for i, j in zip(row_ind[:k], col_ind[:k])]

def _redistribution_delay_us(current: Placement, model, donor_id: int, receiver_id: int) -> float:

    donor_pos = next(p.anchor for p in current.patches if p.qubit == donor_id)
    receiver_pos = next(p.anchor for p in current.patches if p.qubit == receiver_id)
    distance_um = math.hypot(donor_pos.x - receiver_pos.x, donor_pos.y - receiver_pos.y)
    return (15/8) * (distance_um / model.v_xy_um_per_us)

def _resolve_magic_requests(current: Placement, device: Device, schedule: Schedule,
                            factories: list[Factory], qubits: list[int]) -> Placement:

    """
    Event-driven resolution of magic request in stage (FT-Weave)
    1. Factories run parallel check stages
    2. Move spare successes to factories without success
    3. Attempt escape and deliver if success
    """

    if not factories: raise ValueError("No factories")

    ent_zones = [z for z in device.zones if z.kind == "entanglement"]
    if not ent_zones:
        raise ValueError("Device has no entanglement zone")
    ent_zone = ent_zones[0]

    reference_anchor = next(p.anchor for p in current.patches if p.qubit == qubits[0])
    for f in factories:
        current = _ensure_placed(current, ent_zone, f.resource_id, f.distance, reference_anchor)

    by_id = {f.resource_id: f for f in factories}
    remaining = list(qubits)
    waiting_ids = {f.resource_id for f in factories} # running check stage
    busy_escape_us: dict[int, float] = {} # time escape finishes
    pending_escape_outcome: dict[int, bool] = {} # escape success
    wait_for_rus_ids: list[int] = [] # holding T

    while remaining:

        for fid, q in _match_by_distance(current, wait_for_rus_ids, remaining):
            current = _teleport_consume(current, device, schedule, fid, q)
            current = _return_to_zone(current, schedule, ent_zone, fid)
            wait_for_rus_ids.remove(fid)
            remaining.remove(q)

            if by_id[fid].rng.random() >= schedule.model.cultivation_teleportation_success_prob:
                data_atoms = _data_atom_ids(current, q)
                duration = schedule.model.single_qubit_duration_us
                start_us = schedule._advance(data_atoms, duration)
                schedule.Ng1 += len(data_atoms)
                schedule.ops.append(ScheduleOp(
                    kind="correction_gate", start_us=start_us, duration_us=duration,
                    targets=tuple(data_atoms), detail={"qubits": [q]}
                ))

            by_id[fid].restart(now_us=schedule.clock_us)
            waiting_ids.add(fid)

        if not remaining: break

        next_cands = []
        if waiting_ids:
            next_cands.append(min(by_id[fid].check_ready_us for fid in waiting_ids))
        if busy_escape_us:
            next_cands.append(min(busy_escape_us.values()))

        if not next_cands:
            raise ValueError("No pending check or escape attempts")

        next_us = min(next_cands)
        wait = next_us - schedule.clock_us
        if wait > 0:
            start_us = schedule._advance([], wait)
            schedule.ops.append(ScheduleOp(
                kind="await_resource", start_us=start_us, duration_us=wait,
                targets=(), detail={"qubits": list(remaining)}
            ))

        due_ids = [fid for fid in waiting_ids if by_id[fid].check_ready_us <= schedule.clock_us]
        if due_ids:
            for fid in due_ids:
                by_id[fid].simulate_check_stage()
                waiting_ids.discard(fid)

            receivers = [fid for fid in due_ids if not by_id[fid].passed_check()]
            donor_units = [fid for fid in due_ids for _ in range(by_id[fid].num_spare_successes())]

            for recv_fid, donor_fid in _match_by_distance(current, receivers, donor_units):

                delay_us = _redistribution_delay_us(current, schedule.model, donor_fid, recv_fid)
                start_us = schedule._advance([], delay_us)
                schedule.ops.append(ScheduleOp(
                    kind="redistribution_delay", start_us=start_us, duration_us=delay_us,
                    targets=(), detail={"donor": donor_fid, "receiver": recv_fid}
                ))
                by_id[donor_fid].consume_one_success()
                by_id[recv_fid].receive_donated_success()

            for fid in due_ids:
                f = by_id[fid]
                if f.passed_check():
                    success, duration_us = f.attempt_escape()
                    busy_escape_us[fid] = schedule.clock_us + duration_us
                    pending_escape_outcome[fid] = success
                else:
                    f.restart(now_us=schedule.clock_us)
                    waiting_ids.add(fid)

        finished_escapes = [fid for fid, t in busy_escape_us.items() if t <= schedule.clock_us]
        for fid in finished_escapes:
            del busy_escape_us[fid]
            success = pending_escape_outcome.pop(fid)
            if success:
                wait_for_rus_ids.append(fid)
            else:
                by_id[fid].restart(now_us=schedule.clock_us)
                waiting_ids.add(fid)

    return current