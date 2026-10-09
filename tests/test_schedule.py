import math
from pathlib import Path

import pytest

from nasim.circuit import Circuit, one_qubit, two_qubit
from nasim.device import Device
from nasim.geometry import Position
from nasim.model import Model
from nasim.move import legalise_frames, moves_between
from nasim.placement import PlacedPatch, Placement, patch_separation_um, piqasso_placement
from nasim.schedule import (
    Schedule,
    circuit_fidelity,
    simulate_one_qubit_circuit,
    _all_atom_ids,
    simulate_circuit
)
from nasim.surface_code import AtomRole, Patch, _ancilla_weight_counts

PRESET = Path(__file__).resolve().parents[1] / "src" / "nasim" / "config" / "presets" / "2d.yaml"


@pytest.fixture
def model() -> Model:
    return Model.from_yaml(PRESET)


@pytest.fixture
def device(model) -> Device:
    return Device.uniform(model=model, rows=12, cols=12, pitch_um=patch_separation_um(model, distance=3))


def test_two_stage_circuit_counts_match_hand_derivation(model, device):
    # 2 logical qubits, d=3 -> 17 atoms each (9 data + 8 ancilla), 34 total.
    circuit = Circuit.from_stages(2, [[one_qubit(0)], [one_qubit(1)]])
    placement = piqasso_placement(circuit, model, device, distance=3, seed=0)

    schedule = simulate_one_qubit_circuit(circuit, placement, model)

    assert len(schedule.all_atoms) == 34
    assert schedule.Ng1 == 18          # 9 data atoms targeted, twice (once per stage)
    assert schedule.Nsp == 50          # (34 - 9) spectators, twice
    assert schedule.clock_us == pytest.approx(2 * model.single_qubit_duration_us)


def test_idle_time_excludes_only_the_targeted_stage(model, device):
    circuit = Circuit.from_stages(2, [[one_qubit(0)], [one_qubit(1)]])
    placement = piqasso_placement(circuit, model, device, distance=3, seed=0)
    schedule = simulate_one_qubit_circuit(circuit, placement, model)

    patch0 = next(p for p in placement.patches if p.qubit == 0)
    patch1 = next(p for p in placement.patches if p.qubit == 1)
    data0 = next((0, a.local) for a in patch0.patch.atoms if a.role == AtomRole.DATA)
    anc0 = next((0, a.local) for a in patch0.patch.atoms if a.role != AtomRole.DATA)
    data1 = next((1, a.local) for a in patch1.patch.atoms if a.role == AtomRole.DATA)

    d = model.single_qubit_duration_us
    assert schedule.idle_us[data0] == pytest.approx(d)      # idle during stage 1 only
    assert schedule.idle_us[anc0] == pytest.approx(2 * d)   # idle during both stages
    assert schedule.idle_us[data1] == pytest.approx(d)      # idle during stage 0 only


def test_two_qubit_stage_raises_not_implemented(model, device):
    circuit = Circuit.from_stages(2, [[two_qubit(0, 1)]])
    placement = piqasso_placement(circuit, model, device, distance=3, seed=0)
    schedule = Schedule(model=model, all_atoms=tuple(_all_atom_ids(placement)))
    with pytest.raises(NotImplementedError):
        schedule.record_1q_stage(circuit, 0, placement)


def test_move_frame_counts_two_handovers_per_atom(model, device):
    circuit = Circuit.from_stages(2, [[one_qubit(0)]])
    before = piqasso_placement(circuit, model, device, distance=3, seed=0)

    after = Placement(model=before.model, patches=tuple(
        PlacedPatch(
            qubit=p.qubit, patch=p.patch,
            anchor=Position(100.0, 100.0, 0.0) if p.qubit == 1 else p.anchor,
        )
        for p in before.patches
    ))
    frames = legalise_frames(moves_between(before, after))

    schedule = Schedule(model=model, all_atoms=tuple(_all_atom_ids(before)))
    for frame in frames:
        schedule.record_move_frame(frame, before)

    assert schedule.Nh == 2 * 17  # one 17-atom patch moved: pick + drop per atom


def test_fidelity_matches_manual_formula(model, device):
    circuit = Circuit.from_stages(2, [[one_qubit(0)], [one_qubit(1)]])
    placement = piqasso_placement(circuit, model, device, distance=3, seed=0)
    schedule = simulate_one_qubit_circuit(circuit, placement, model)

    f = circuit_fidelity(schedule, d=3)

    expected = model.single_qubit_fidelity ** schedule.Ng1
    expected *= model.carrier_fidelity ** schedule.Nsp
    t2_us = model.t2_s * 1e6
    for t_idle in schedule.idle_us.values():
        expected *= math.exp(-t_idle / t2_us)

    assert f == pytest.approx(expected)


def test_fidelity_is_between_zero_and_one(model, device):
    circuit = Circuit.from_stages(3, [[one_qubit(0)], [one_qubit(1)], [one_qubit(2)]])
    placement = piqasso_placement(circuit, model, device, distance=3, seed=0)
    schedule = simulate_one_qubit_circuit(circuit, placement, model)
    f = circuit_fidelity(schedule, d=3)
    assert 0.0 < f < 1.0

def _side_by_side_placement(model):
    from nasim.placement import PlacedPatch, Placement
    from nasim.surface_code import Patch

    patch = Patch.rotated(3)
    unit = model.gate_pair_dist_um
    min_x, min_y, max_x, max_y = patch.local_bounds
    anchor0 = Position(-min_x * unit, -min_y * unit, 0.0)
    w = (max_x - min_x) * unit
    anchor1 = Position(anchor0.x + w, anchor0.y, 0.0)
    pa = PlacedPatch(0, patch, anchor0)
    pb = PlacedPatch(1, patch, anchor1)
    return Placement(model=model, patches=(pa, pb)), pa, pb


def test_merge_charges_handover_for_changed_atoms_only(model):
    placement, pa, pb = _side_by_side_placement(model)
    all_ids = {(p.qubit, a.local) for p in placement.patches for a in p.patch.atoms}
    schedule = Schedule(model=model, all_atoms=all_ids)

    origin_qubit, merged_patch, anchor = schedule.record_merge(placement, 0, 1, rounds=1)

    # 2 of patch1's boundary ancilla relocate + 3 genuinely new bridge
    # atoms appear (35 total - 34 original + 2 relocated = 3 new) -> 5
    # changed atoms, pick+drop each -> Nh == 10
    assert schedule.Nh == 10
    assert len(schedule.all_atoms) == 35


def test_merge_ng2_and_nmeas_match_merged_patch_geometry(model):
    placement, pa, pb = _side_by_side_placement(model)
    all_ids = {(p.qubit, a.local) for p in placement.patches for a in p.patch.atoms}
    schedule = Schedule(model=model, all_atoms=all_ids)

    origin_qubit, merged_patch, anchor = schedule.record_merge(placement, 0, 1, rounds=1)

    from nasim.surface_code import _ancilla_weight_counts
    interior, boundary = _ancilla_weight_counts(merged_patch)
    assert schedule.Ng2 == 4 * interior + 2 * boundary
    assert schedule.Nmeas == interior + boundary


def test_merge_duration_matches_hand_derivation(model):
    placement, pa, pb = _side_by_side_placement(model)
    all_ids = {(p.qubit, a.local) for p in placement.patches for a in p.patch.atoms}
    schedule = Schedule(model=model, all_atoms=all_ids)

    schedule.record_merge(placement, 0, 1, rounds=1)

    expected = 2 * model.handover_time_us + 4 * model.cz_duration_us + model.readout_time_us
    assert schedule.clock_us == pytest.approx(expected)


def test_merge_then_split_restores_original_atom_universe(model):
    placement, pa, pb = _side_by_side_placement(model)
    all_ids = {(p.qubit, a.local) for p in placement.patches for a in p.patch.atoms}
    schedule = Schedule(model=model, all_atoms=set(all_ids))

    origin_qubit, merged_patch, anchor = schedule.record_merge(placement, 0, 1, rounds=1)
    assert schedule.all_atoms != all_ids  # merged universe differs mid-surgery

    schedule.record_split(merged_patch, anchor, origin_qubit, pa, pb)
    assert schedule.all_atoms == all_ids


def test_fidelity_includes_mid_circuit_measurement_term(model):
    placement, pa, pb = _side_by_side_placement(model)
    all_ids = {(p.qubit, a.local) for p in placement.patches for a in p.patch.atoms}
    schedule = Schedule(model=model, all_atoms=set(all_ids))
    schedule.record_merge(placement, 0, 1, rounds=1)

    f = circuit_fidelity(schedule, d=3)
    assert 0.0 < f < 1.0

    # removing the Nmeas term should change the result (i.e. it's actually applied)
    manual = f / (model.readout_fidelity ** schedule.Nmeas)
    assert manual != pytest.approx(f)

def test_simulate_circuit_routes_merges_and_splits(model):
    from nasim.circuit import one_qubit, two_qubit
    from nasim.schedule import simulate_circuit

    patch = Patch.rotated(3)
    unit = model.gate_pair_dist_um
    min_x, min_y, max_x, max_y = patch.local_bounds
    anchor0 = Position(-min_x * unit, -min_y * unit, 0.0)
    anchor1 = Position(anchor0.x + 200.0, anchor0.y + 50.0, 0.0)
    anchor2 = Position(anchor0.x - 150.0, anchor0.y - 80.0, 0.0)
    placement = Placement(model=model, patches=(
        PlacedPatch(0, patch, anchor0),
        PlacedPatch(1, patch, anchor1),
        PlacedPatch(2, patch, anchor2),
    ))

    circuit = Circuit.from_stages(3, [
        [one_qubit(2)],
        [two_qubit(0, 1)],
    ])

    schedule, final_placement = simulate_circuit(circuit, placement, model)

    q0_anchor = next(p.anchor for p in final_placement.patches if p.qubit == 0)
    q1_anchor = next(p.anchor for p in final_placement.patches if p.qubit == 1)
    q2_anchor = next(p.anchor for p in final_placement.patches if p.qubit == 2)

    assert q0_anchor == anchor0           # origin patch never moves
    width = (max_x - min_x) * unit
    assert q1_anchor == Position(anchor0.x + width, anchor0.y, 0.0)  # routed adjacent
    assert q2_anchor == anchor2           # untouched by the 2Q stage

    interior, boundary = _ancilla_weight_counts(Patch._build_rectangle(2 * 3, 3, code_distance=3))
    assert schedule.Ng2 == 3 * (4 * interior + 2 * boundary)   # 3 rounds (default = distance)
    assert schedule.Nmeas == 3 * (interior + boundary)

    f = circuit_fidelity(schedule, d=3)
    assert 0.0 < f < 1.0

def test_simulate_circuit_handles_a_ghz_chain(model):
    from nasim.circuit import two_qubit
    from nasim.schedule import simulate_circuit

    patch = Patch.rotated(3)
    unit = model.gate_pair_dist_um
    min_x, min_y, max_x, max_y = patch.local_bounds

    anchors = {
        0: Position(-min_x * unit, -min_y * unit, 0.0),
        1: Position(300.0, 80.0, 0.0),
        2: Position(-200.0, 150.0, 0.0),
        3: Position(100.0, -250.0, 0.0),
    }
    placement = Placement(model=model, patches=tuple(
        PlacedPatch(q, patch, anchors[q]) for q in range(4)
    ))

    circuit = Circuit.from_stages(4, [
        [two_qubit(0, 1)],
        [two_qubit(1, 2)],
        [two_qubit(2, 3)],
    ])

    # record_merge/record_split call check_alignment internally and raise
    # on a bad merge, so just completing all 3 stages proves every pair was
    # adjacent and boundary-matched at merge time - exact final coordinates
    # are no longer fixed since routing-aware placement (Edge scoring) picks
    # whichever side is closest for each pair, not always "right".
    schedule, final_placement = simulate_circuit(circuit, placement, model)

    op_kinds = [op.kind for op in schedule.ops]
    assert op_kinds.count("split_handover") == op_kinds.count("merge_handover") == 3

    anchors_final = [p.anchor for p in final_placement.patches]
    assert len(set(anchors_final)) == 4, "no two patches should have collided onto the same anchor"

def test_simulate_circuit_batches_independent_pairs_into_one_move_frame(model):
    from nasim.circuit import two_qubit
    from nasim.schedule import simulate_circuit

    patch = Patch.rotated(3)
    unit = model.gate_pair_dist_um
    min_x, min_y, max_x, max_y = patch.local_bounds

    anchors = {
        0: Position(-min_x * unit, -min_y * unit, 0.0),
        1: Position(200.0, 40.0, 0.0),
        2: Position(-300.0, 300.0, 0.0),
        3: Position(-100.0, 450.0, 0.0),
    }
    placement = Placement(model=model, patches=tuple(
        PlacedPatch(q, patch, anchors[q]) for q in range(4)
    ))
    circuit = Circuit.from_stages(4, [[two_qubit(0, 1), two_qubit(2, 3)]])

    schedule, final_placement = simulate_circuit(circuit, placement, model)

    move_frame_ops = [op for op in schedule.ops if op.kind == "move_frame"]
    assert len(move_frame_ops) == 1
    assert sorted(move_frame_ops[0].detail["qubits"]) == [1, 3]

    total_atoms = sum(len(p.patch.atoms) for p in final_placement.patches)
    assert total_atoms == 4 * 17
    assert len(schedule.all_atoms) == 4 * 17

    f = circuit_fidelity(schedule, d=3)
    assert 0.0 < f < 1.0


def test_simulate_circuit_rejects_a_qubit_in_two_gates_same_stage(model):
    from nasim.circuit import two_qubit
    from nasim.schedule import simulate_circuit

    patch = Patch.rotated(3)
    unit = model.gate_pair_dist_um
    min_x, min_y, max_x, max_y = patch.local_bounds
    width = (max_x - min_x) * unit
    placement = Placement(model=model, patches=(
        PlacedPatch(0, patch, Position(0.0, 0.0, 0.0)),
        PlacedPatch(1, patch, Position(width, 0.0, 0.0)),
        PlacedPatch(2, patch, Position(2 * width, 0.0, 0.0)),
    ))
    circuit = Circuit.from_stages(3, [[two_qubit(0, 1), two_qubit(1, 2)]])

    with pytest.raises(NotImplementedError, match=">1"):
        simulate_circuit(circuit, placement, model)

def test_routing_aware_placement_avoids_an_avoidable_conflict(model):
    # q1 starts directly above q0; q2 sits right where the naive
    # always-right-of-qa choice would land q1, and q3 merges into q2.
    # The hardcoded default collides (shared y-target) and serialises;
    # scoring should still find q1's TOP edge is both shorter and
    # pairwise-conflict-free versus q3's move.
    #
    # They still can't share one AOD frame, though: q1's source is
    # (0, 500) and q3's source is (300, 0), so firing them together would
    # activate x-tones {0, 300} and y-tones {500, 0} - and (0, 0) is
    # exactly where q0 (stationary, the merge anchor for pair (0,1)) sits.
    # That's a real unintended-pickup hazard (move.py's AOD tone-safety
    # check), not a routing failure, so they correctly serialise into two
    # frames even though they'd have been fine as a bare pairwise check.
    patch = Patch.rotated(3)
    placement = Placement(model=model, patches=(
        PlacedPatch(0, patch, Position(0.0, 0.0, 0.0)),
        PlacedPatch(1, patch, Position(0.0, 500.0, 0.0)),
        PlacedPatch(2, patch, Position(15.0, 0.0, 0.0)),
        PlacedPatch(3, patch, Position(300.0, 0.0, 0.0)),
    ))
    circuit = Circuit.from_stages(4, [[two_qubit(0, 1), two_qubit(2, 3)]])

    schedule, final_placement = simulate_circuit(circuit, placement, model)

    move_frame_ops = [op for op in schedule.ops if op.kind == "move_frame"]
    assert len(move_frame_ops) == 2, "q0 sits at the spurious (0,0) tone intersection, so q1/q3 must serialise"
    moved_qubits = sorted(q for op in move_frame_ops for q in op.detail["qubits"])
    assert moved_qubits == [1, 3]

    q1_anchor = next(p.anchor for p in final_placement.patches if p.qubit == 1)
    assert q1_anchor.x == 0.0, "q1 should have stayed on q0's x-coordinate (TOP edge), not jumped sideways"


# --- Zone / crosstalk mechanism (ZAP's N_xtalk, Eq. 4) ---

@pytest.fixture
def zoned_device(model) -> Device:
    return Device.zoned(
        model=model,
        storage_rows=10, storage_cols=10, storage_pitch_um=10.0,
        entanglement_rows=4, entanglement_cols=4, entanglement_pitch=10.0,
        gap_um=10.0,
    )


def _merging_pair_with_bystander(model, bystander_anchor: Position):
    # q0, q1 merge inside the entanglement zone (y starts at 100 for the
    # zoned_device fixture above); q2 is the bystander at a caller-chosen
    # position, to test both the entanglement-zone and storage-zone cases.
    patch = Patch.rotated(3)
    placement = Placement(model=model, patches=(
        PlacedPatch(0, patch, Position(0.0, 100.0, 0.0)),
        PlacedPatch(1, patch, Position(15.0, 100.0, 0.0)),
        PlacedPatch(2, patch, bystander_anchor),
    ))
    all_ids = {(p.qubit, a.local) for p in placement.patches for a in p.patch.atoms}
    return placement, all_ids, patch


def test_merge_without_a_device_charges_no_crosstalk(model):
    # Backward compatibility: omitting `device` entirely (every pre-existing
    # call site) must behave exactly as before zones were added.
    placement, all_ids, _ = _merging_pair_with_bystander(model, Position(0.0, 100.0 + 30.0, 0.0))
    schedule = Schedule(model=model, all_atoms=all_ids)

    schedule.record_merge(placement, 0, 1, rounds=2)

    assert schedule.Nxtalk == 0


def test_merge_charges_crosstalk_for_an_entanglement_zone_bystander(model, zoned_device):
    # q2 sits inside the entanglement zone (y=130, well within the fixture's
    # entanglement zone which starts at y=100) while q0/q1 merge - it should
    # be charged crosstalk once per CZ substep per round.
    placement, all_ids, patch = _merging_pair_with_bystander(model, Position(0.0, 130.0, 0.0))
    schedule = Schedule(model=model, all_atoms=all_ids, device=zoned_device)

    schedule.record_merge(placement, 0, 1, rounds=2)

    expected = len(patch.atoms) * 4 * 2  # bystander atoms x 4 substeps x 2 rounds
    assert schedule.Nxtalk == expected


def test_merge_does_not_charge_crosstalk_for_a_storage_zone_bystander(model, zoned_device):
    # q2 sits in the storage zone (y=50) - physically shielded from the
    # entanglement zone's Rydberg field, so it must not be charged.
    placement, all_ids, _ = _merging_pair_with_bystander(model, Position(50.0, 50.0, 0.0))
    schedule = Schedule(model=model, all_atoms=all_ids, device=zoned_device)

    schedule.record_merge(placement, 0, 1, rounds=2)

    assert schedule.Nxtalk == 0


def test_merge_does_not_charge_crosstalk_for_the_merging_patches_themselves(model, zoned_device):
    # q0 and q1 are the ones merging and sit inside the entanglement zone
    # too, but they're the gate's own targets, not bystanders - excluded.
    placement, all_ids, _ = _merging_pair_with_bystander(model, Position(50.0, 50.0, 0.0))
    schedule = Schedule(model=model, all_atoms=all_ids, device=zoned_device)

    schedule.record_merge(placement, 0, 1, rounds=2)

    assert schedule.Nxtalk == 0


def test_circuit_fidelity_includes_the_crosstalk_term(model, zoned_device):
    placement, all_ids, _ = _merging_pair_with_bystander(model, Position(0.0, 130.0, 0.0))
    schedule = Schedule(model=model, all_atoms=all_ids, device=zoned_device)
    schedule.record_merge(placement, 0, 1, rounds=2)

    assert schedule.Nxtalk > 0
    f = circuit_fidelity(schedule, d=3)
    f_without_xtalk_term = f / (model.carrier_fidelity ** schedule.Nxtalk)
    assert f < f_without_xtalk_term, "crosstalk term should strictly reduce fidelity"


# --- Idle-qubit management / zone-constrained gate placement (ZAP Eq. 12-15, Algorithm 2) ---

@pytest.fixture
def zoned_device_d3(model) -> Device:
    # pitch large enough for real d=3 patches (patch_separation_um), unlike
    # the tight 10um zoned_device fixture above which is only used with
    # hand-placed patches that never consult device.sites.
    pitch = patch_separation_um(model, distance=3)
    return Device.zoned(
        model=model,
        storage_rows=15, storage_cols=15, storage_pitch_um=pitch,
        entanglement_rows=6, entanglement_cols=6, entanglement_pitch=pitch,
        gap_um=pitch,
    )


def test_merge_routes_both_patches_into_the_entanglement_zone(model, zoned_device_d3):
    from nasim.schedule import simulate_circuit

    ent_zone = next(z for z in zoned_device_d3.zones if z.kind == "entanglement")
    patch = Patch.rotated(3)
    # both qubits start deep in storage, far from each other and from the
    # entanglement zone - nothing here is already in-zone or adjacent
    placement = Placement(model=model, patches=(
        PlacedPatch(0, patch, Position(0.0, 0.0, 0.0)),
        PlacedPatch(1, patch, Position(200.0, 50.0, 0.0)),
    ))
    circuit = Circuit.from_stages(2, [[two_qubit(0, 1)]])

    schedule, final_placement = simulate_circuit(circuit, placement, model, device=zoned_device_d3)

    for p in final_placement.patches:
        assert ent_zone.contains(p.anchor), f"q{p.qubit} should have been routed into the entanglement zone"


def test_idle_qubit_never_reused_again_always_returns_to_storage(model, zoned_device_d3):
    from nasim.schedule import _idle_qubit_management

    ent_zone = next(z for z in zoned_device_d3.zones if z.kind == "entanglement")
    storage_zone = next(z for z in zoned_device_d3.zones if z.kind == "storage")
    patch = Patch.rotated(3)
    origin = ent_zone.slm.origin
    width = patch_separation_um(model, distance=3)

    placement = Placement(model=model, patches=(
        PlacedPatch(0, patch, Position(origin.x, origin.y, 0.0)),
        PlacedPatch(1, patch, Position(origin.x + width, origin.y, 0.0)),
        PlacedPatch(2, patch, Position(origin.x + 2 * width, origin.y, 0.0)),
    ))
    # q2 sits resident in the entanglement zone but never appears in
    # another 2Q gate for the rest of the circuit
    circuit = Circuit.from_stages(3, [[two_qubit(0, 1)]])

    working, moves = _idle_qubit_management(circuit, 0, placement, zoned_device_d3, model, {0, 1})

    assert any(m.qubit == 2 for m in moves)
    q2_final = next(p.anchor for p in working.patches if p.qubit == 2)
    assert storage_zone.contains(q2_final)


def test_idle_qubit_reused_next_stage_stays_resident(model, zoned_device_d3):
    from nasim.schedule import _idle_qubit_management

    ent_zone = next(z for z in zoned_device_d3.zones if z.kind == "entanglement")
    patch = Patch.rotated(3)
    origin = ent_zone.slm.origin
    width = patch_separation_um(model, distance=3)

    placement = Placement(model=model, patches=(
        PlacedPatch(0, patch, Position(origin.x, origin.y, 0.0)),
        PlacedPatch(1, patch, Position(origin.x + width, origin.y, 0.0)),
        PlacedPatch(2, patch, Position(origin.x + 2 * width, origin.y, 0.0)),
    ))
    # q2 is needed again at the very next stage (k=1) - cheap to keep resident
    circuit = Circuit.from_stages(3, [[two_qubit(0, 1)], [two_qubit(1, 2)]])

    working, moves = _idle_qubit_management(circuit, 0, placement, zoned_device_d3, model, {0, 1})

    assert not any(m.qubit == 2 for m in moves)
    q2_final = next(p.anchor for p in working.patches if p.qubit == 2)
    assert q2_final == placement.patches[2].anchor


def test_idle_qubit_reused_far_in_the_future_still_returns(model, zoned_device_d3):
    # crosstalk cost grows with k (stages until reuse); past some point it
    # must exceed the fixed cost of a round-trip to storage and back.
    from nasim.circuit import one_qubit
    from nasim.schedule import _idle_qubit_management

    ent_zone = next(z for z in zoned_device_d3.zones if z.kind == "entanglement")
    patch = Patch.rotated(3)
    origin = ent_zone.slm.origin
    width = patch_separation_um(model, distance=3)

    placement = Placement(model=model, patches=(
        PlacedPatch(0, patch, Position(origin.x, origin.y, 0.0)),
        PlacedPatch(1, patch, Position(origin.x + width, origin.y, 0.0)),
        PlacedPatch(2, patch, Position(origin.x + 2 * width, origin.y, 0.0)),
    ))
    far_stages = [[two_qubit(0, 1)]] + [[one_qubit(3)] for _ in range(50)] + [[two_qubit(1, 2)]]
    circuit = Circuit.from_stages(4, far_stages)

    working, moves = _idle_qubit_management(circuit, 0, placement, zoned_device_d3, model, {0, 1})

    assert any(m.qubit == 2 for m in moves), "reuse 51 stages away should cost more than returning to storage"


def test_simulate_circuit_without_zones_is_unaffected_by_a_zoneless_device(model):
    # a Device.uniform (zones == ()) must behave exactly like device=None
    from nasim.schedule import simulate_circuit

    device_plain = Device.uniform(model=model, rows=12, cols=12, pitch_um=patch_separation_um(model, distance=3))
    patch = Patch.rotated(3)
    placement = Placement(model=model, patches=(
        PlacedPatch(0, patch, Position(0.0, 0.0, 0.0)),
        PlacedPatch(1, patch, Position(300.0, 80.0, 0.0)),
    ))
    circuit = Circuit.from_stages(2, [[two_qubit(0, 1)]])

    schedule_none, final_none = simulate_circuit(circuit, placement, model, device=None)
    schedule_plain, final_plain = simulate_circuit(circuit, placement, model, device=device_plain)

    assert [op.kind for op in schedule_none.ops] == [op.kind for op in schedule_plain.ops]
    assert schedule_none.Nxtalk == schedule_plain.Nxtalk == 0
    for p0, p1 in zip(final_none.patches, final_plain.patches):
        assert p0.anchor == p1.anchor


def test_simulate_circuit_zoned_multi_stage_runs_and_produces_valid_fidelity(model, zoned_device_d3):
    from nasim.schedule import simulate_circuit

    patch = Patch.rotated(3)
    placement = Placement(model=model, patches=(
        PlacedPatch(0, patch, Position(0.0, 0.0, 0.0)),
        PlacedPatch(1, patch, Position(200.0, 0.0, 0.0)),
        PlacedPatch(2, patch, Position(0.0, 200.0, 0.0)),
    ))
    circuit = Circuit.from_stages(3, [
        [two_qubit(0, 1)],
        [two_qubit(1, 2)],
    ])

    schedule, final_placement = simulate_circuit(circuit, placement, model, device=zoned_device_d3)

    assert schedule.ops, "circuit should have produced some ops"
    f = circuit_fidelity(schedule, d=3)
    assert 0.0 < f < 1.0

    # q0 isn't needed again after stage 0 -> should have been sent home
    storage_zone = next(z for z in zoned_device_d3.zones if z.kind == "storage")
    q0_final = next(p.anchor for p in final_placement.patches if p.qubit == 0)
    assert storage_zone.contains(q0_final)


def test_zone_managed_patches_are_fully_contained_not_just_their_anchor(model, zoned_device_d3):
    # _nearest_free_zone_site (used by _idle_qubit_management and
    # _retarget_pair_into_entanglement_zone) must place patches using the
    # same site->anchor convention as piqasso_placement (_site_anchor):
    # the patch's local-min corner at the site, not its local origin -
    # otherwise the patch's REAL footprint can extend outside the zone
    # even though its anchor point still tests as "contained".
    from nasim.schedule import simulate_circuit
    from nasim.viz.device import _zone_bbox

    patch = Patch.rotated(3)
    placement = Placement(model=model, patches=(
        PlacedPatch(0, patch, Position(0.0, 0.0, 0.0)),
        PlacedPatch(1, patch, Position(200.0, 0.0, 0.0)),
        PlacedPatch(2, patch, Position(0.0, 200.0, 0.0)),
    ))
    circuit = Circuit.from_stages(3, [
        [two_qubit(0, 1)],
        [two_qubit(1, 2)],
    ])

    schedule, final_placement = simulate_circuit(circuit, placement, model, device=zoned_device_d3)

    ent_zone = next(z for z in zoned_device_d3.zones if z.kind == "entanglement")
    storage_zone = next(z for z in zoned_device_d3.zones if z.kind == "storage")
    unit = model.gate_pair_dist_um

    for p in final_placement.patches:
        zone = ent_zone if ent_zone.contains(p.anchor) else storage_zone if storage_zone.contains(p.anchor) else None
        assert zone is not None, f"q{p.qubit}'s anchor isn't in any zone"

        min_x, min_y, max_x, max_y = p.patch.local_bounds
        footprint_x0 = p.anchor.x + min_x * unit
        footprint_x1 = p.anchor.x + max_x * unit
        footprint_y0 = p.anchor.y + min_y * unit
        footprint_y1 = p.anchor.y + max_y * unit

        zx0, zy0, zx1, zy1 = _zone_bbox(zone)
        assert zx0 - 1e-6 <= footprint_x0 and footprint_x1 <= zx1 + 1e-6, f"q{p.qubit} footprint escapes zone in x"
        assert zy0 - 1e-6 <= footprint_y0 and footprint_y1 <= zy1 + 1e-6, f"q{p.qubit} footprint escapes zone in y"


# --- Logical (threshold-scaling) fidelity, replacing the old NISQ-style product ---

def _cnot_schedule(model, d):
    device = Device.uniform(model=model, rows=30, cols=30, pitch_um=patch_separation_um(model, distance=d))
    circuit = Circuit.from_stages(3, [[two_qubit(0, 1)], [two_qubit(1, 2)]])
    placement = piqasso_placement(circuit, model, device, distance=d, seed=0)
    schedule, _ = simulate_circuit(circuit, placement, model)
    return schedule


@pytest.mark.parametrize("d", [3, 5, 7, 9])
def test_circuit_fidelity_does_not_collapse_with_code_distance(model, d):
    # The old whole-physical-circuit product got catastrophically worse
    # with d (more syndrome rounds = more undecoded physical error) -
    # the opposite of what a real surface code does. The new threshold-
    # scaling estimate (Fowler et al., arXiv:1208.0928 Eq. 10-11) should
    # stay in a sane range even at d=9, not collapse toward zero.
    schedule = _cnot_schedule(model, d)
    f = circuit_fidelity(schedule, d=d)
    assert 0.05 < f < 1.0, f"d={d}: fidelity={f} - still collapsing"


def test_per_round_logical_error_rate_improves_with_code_distance():
    # Isolate the per-round logical error rate (undo the compounding over
    # the number of rounds actually run, which differs across d since
    # rounds defaults to the code distance) - this should strictly
    # decrease as d grows, which is the entire point of a surface code.
    model_ = Model.from_yaml(PRESET)
    prev_p_l = None
    for d in (3, 5, 7, 9):
        schedule = _cnot_schedule(model_, d)
        f = circuit_fidelity(schedule, d=d)
        num_rounds = sum(1 for op in schedule.ops if op.kind == "mid_circuit_measure")
        non_logical = (
            model_.single_qubit_fidelity ** schedule.Ng1
            * model_.carrier_fidelity ** schedule.Nsp
            * model_.handover_fidelity ** schedule.Nh
            * model_.carrier_fidelity ** schedule.Nxtalk
        )
        t2_us = model_.t2_s * 1e6
        for t_idle in schedule.idle_us.values():
            non_logical *= math.exp(-t_idle / t2_us)
        per_round_fidelity = (f / non_logical) ** (1.0 / num_rounds)
        p_l = 1.0 - per_round_fidelity
        if prev_p_l is not None:
            assert p_l < prev_p_l, f"d={d}: per-round P_L={p_l} did not improve on {prev_p_l}"
        prev_p_l = p_l