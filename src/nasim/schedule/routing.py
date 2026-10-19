from __future__ import annotations

import math

from nasim.device import Device
from nasim.geometry import Position
from nasim.move import Move, conflict_reason
from nasim.placement import Placement, PlacedPatch
from nasim.surface_code import Patch, Edge

_CAND_EDGES = (Edge.RIGHT, Edge.LEFT, Edge.TOP, Edge.BOTTOM)

_CONFLICT_COST = {
    "crossing": 3.0,
    "shared_target": 2.0,
    "shared_source": 0.5 # can use parking here
}

def _cand_anchor(pa: PlacedPatch, edge: Edge, width: float, height: float) -> Position:

    if edge is Edge.RIGHT:
        return Position(pa.anchor.x + width, pa.anchor.y, pa.anchor.z)
    elif edge is Edge.LEFT:
        return Position(pa.anchor.x - width, pa.anchor.y, pa.anchor.z)
    elif edge is Edge.TOP:
        return Position(pa.anchor.x, pa.anchor.y + height, pa.anchor.z)
    return Position(pa.anchor.x, pa.anchor.y - height, pa.anchor.z)

def _conflict_cost(move: Move, committed: list[Move]) -> float:

    return sum(
        _CONFLICT_COST.get(reason[1], 0.0)
        for other in committed
        if (reason := conflict_reason(move, other)) is not None
    )

def _site_is_free(target_anchor: Position, radius: float, placement: Placement, exclude_qubit: int,
                  *, eps: float = 1e-6) -> bool:

    for p in placement.patches:
        if p.qubit == exclude_qubit: continue

        other_radius = p.patch.radius_um(placement.model.gate_pair_dist_um)
        d = math.hypot(target_anchor.x - p.anchor.x, target_anchor.y - p.anchor.y)
        if d < radius + other_radius - eps: return False

    return True

def _best_candidate(placement: Placement, fixed: PlacedPatch, moving: PlacedPatch, committed: list[Move],
                    *, lambda_par: float, site_filter) -> tuple[Move | None, float]:

    unit_um = placement.model.gate_pair_dist_um
    min_x, min_y, max_x, max_y = fixed.patch.local_bounds
    width = (max_x - min_x) * unit_um
    height = (max_y - min_y) * unit_um
    radius = moving.patch.radius_um(unit_um)

    best_move, best_score = None, math.inf
    for edge in _CAND_EDGES:
        target_anchor = _cand_anchor(fixed, edge, width, height)
        if not _site_is_free(target_anchor, radius, placement, exclude_qubit=moving.qubit): continue
        if site_filter is not None and not site_filter(target_anchor): continue

        candidate = Move(qubit=moving.qubit, source=moving.anchor, target=target_anchor, clearance_um=radius)

        score = lambda_par * _conflict_cost(candidate, committed) + candidate.distance_um()
        if score < best_score:
            best_score, best_move = score, candidate

    return best_move, best_score

def _site_anchor(site: Position, patch: Patch, unit_um: float) -> Position:

    min_x, min_y, _, _ = patch.local_bounds
    return Position(site.x - min_x * unit_um, site.y - min_y * unit_um, site.z)

def _best_relocated_candidate(placement: Placement, relocate_qubit: int, other_qubit: int,
                              committed: list[Move], *, zone, lambda_par: float,
                              site_filter) -> tuple[Move | None, Move | None, float]:

    p_reloc = next(p for p in placement.patches if p.qubit == relocate_qubit)
    p_other = next(p for p in placement.patches if p.qubit == other_qubit)
    unit_um = placement.model.gate_pair_dist_um
    radius = p_reloc.patch.radius_um(unit_um)

    candidate_sites = sorted(
        zone.sites,
        key=lambda s: math.hypot(s.x - p_reloc.anchor.x, s.y - p_reloc.anchor.y)
    )

    for site in candidate_sites:
        new_anchor = _site_anchor(site, p_reloc.patch, unit_um)
        if not _site_is_free(new_anchor, radius, placement, exclude_qubit=relocate_qubit): continue
        if site_filter is not None and not site_filter(new_anchor): continue

        reloc_move = Move(qubit=relocate_qubit, source=p_reloc.anchor, target=new_anchor, clearance_um=radius)

        relocated_patch = PlacedPatch(qubit=relocate_qubit, patch=p_reloc.patch, anchor=new_anchor)

        touch_move, touch_score = _best_candidate(placement, relocated_patch, p_other, committed + [reloc_move],
                                                  lambda_par=lambda_par, site_filter=site_filter)

        if touch_move is not None:
            return reloc_move, touch_move, reloc_move.distance_um() + touch_score

    return None, None, math.inf


def _retarget_adjacent(placement: Placement, qa: int, qb: int, committed: list[Move], *, lambda_par: float = 25.0,
                       site_filter=None, zone=None) -> Placement:

    pa = next(p for p in placement.patches if p.qubit == qa)
    pb = next(p for p in placement.patches if p.qubit == qb)
    if pa.patch.distance != pb.patch.distance:
        raise ValueError("Patches have different distances")

    move_b, score_b = _best_candidate(placement, pa, pb, committed, lambda_par=lambda_par, site_filter=site_filter)
    move_a, score_a = _best_candidate(placement, pb, pa, committed, lambda_par=lambda_par, site_filter=site_filter)

    reloc_move = None
    if move_b is None and move_a is None and zone is not None:
        reloc_b, touch_b, score_rb = _best_relocated_candidate(placement, qa, qb, committed, 
                                                               zone=zone, lambda_par=lambda_par, site_filter=site_filter)
        reloc_a, touch_a, score_ra = _best_relocated_candidate(placement, qb, qa, committed,
                                                               zone=zone, lambda_par=lambda_par, site_filter=site_filter)

        if touch_b is not None and (touch_a is None or score_rb <= score_ra):
            reloc_move, move_b, score_b = reloc_b, touch_b, score_rb
        elif touch_a is not None:
            reloc_move, move_a, score_a = reloc_a, touch_a, score_ra

    if move_b is None and move_a is None:
        raise ValueError("No valid sites found")

    best_move = move_b if (move_a is None or (move_b is not None and score_b <= score_a)) else move_a

    patches = tuple(
        PlacedPatch(
            qubit=p.qubit, patch=p.patch,
            anchor=(reloc_move.target if reloc_move is not None and p.qubit == reloc_move.qubit
                    else best_move.target if p.qubit == best_move.qubit else p.anchor)
        )
        for p in placement.patches
    )

    moves = [m for m in (reloc_move, best_move) if m is not None]
    return Placement(model=placement.model, patches=patches), moves

def _nearest_free_zone_site(zone, placement: Placement, patch: Patch, unit_um: float, anchor: Position, *,
                            radius: float, exclude_qubit: int) -> Position | None:

    best_site, best_d = None, math.inf
    for site in zone.sites:
        candidate = _site_anchor(site, patch, unit_um)
        if not _site_is_free(candidate, radius, placement, exclude_qubit=exclude_qubit): continue
        d = math.hypot(candidate.x - anchor.x, candidate.y - anchor.y)
        if d < best_d:
            best_d, best_site = d, candidate

    return best_site

def _retarget_pair_into_entanglement_zone(placement: Placement, device: Device, qa: int, qb: int,
                                          committed: list[Move], *, lambda_par: float = 25.0):

    pa = next(p for p in placement.patches if p.qubit == qa)
    pb = next(p for p in placement.patches if p.qubit == qb)
    if pa.patch.distance != pb.patch.distance:
            raise ValueError("Patches have different distances")

    ent_zones = [z for z in device.zones if z.kind == "entanglement"]
    if not ent_zones:
        placement2, moves = _retarget_adjacent(placement, qa, qb, committed, lambda_par=lambda_par)
        return placement2, moves

    ent_zone = ent_zones[0]
    unit_um = placement.model.gate_pair_dist_um
    radius_a = pa.patch.radius_um(unit_um)

    moves: list[Move] = []
    working = placement

    if not ent_zone.contains(pa.anchor):

        target = _nearest_free_zone_site(ent_zone, working, pa.patch, unit_um, pa.anchor, radius=radius_a, exclude_qubit=qa)
        if target is None: raise ValueError("No free Ez site")

        move_a = Move(qubit=qa, source=pa.anchor, target=target, clearance_um=radius_a)
        patches = tuple(
            PlacedPatch(qubit=p.qubit, patch=p.patch, anchor=target if p.qubit == qa else p.anchor)
            for p in working.patches
        )
        working = Placement(model=working.model, patches=patches)
        moves.append(move_a)

    working, moves_b = _retarget_adjacent(working, qa, qb, committed + moves, lambda_par=lambda_par, site_filter=ent_zone.contains, zone=ent_zone)
    moves.extend(moves_b)

    return working, moves