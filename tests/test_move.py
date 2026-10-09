from pathlib import Path

import pytest
import math

from nasim.circuit import Circuit, two_qubit
from nasim.device import Device
from nasim.geometry import Position
from nasim.model import Model
from nasim.move import Move, compatible_2d, legalise_frames, move_duration_us, moves_between
from nasim.placement import patch_separation_um, piqasso_placement

PRESET = Path(__file__).resolve().parents[1] / "src" / "nasim" / "config" / "presets" / "2d.yaml"


@pytest.fixture
def model() -> Model:
    return Model.from_yaml(PRESET)


@pytest.fixture
def device(model) -> Device:
    return Device.uniform(model=model, rows=12, cols=12, pitch_um=patch_separation_um(model, distance=3))


def test_crossing_moves_are_incompatible():
    a = Move(0, Position(0, 0, 0), Position(10, 0, 0))
    b = Move(1, Position(5, 0, 0), Position(2, 0, 0))
    assert not compatible_2d(a, b)


def test_parallel_non_crossing_moves_are_compatible():
    a = Move(0, Position(0, 0, 0), Position(10, 0, 0))
    b = Move(1, Position(5, 0, 0), Position(15, 0, 0))
    assert compatible_2d(a, b)


def test_shared_start_tone_diverging_targets_is_incompatible():
    a = Move(0, Position(5, 0, 0), Position(10, 0, 0))
    b = Move(1, Position(5, 0, 0), Position(20, 0, 0))
    assert not compatible_2d(a, b)


def test_shared_end_tone_different_starts_is_incompatible():
    a = Move(0, Position(0, 0, 0), Position(10, 0, 0))
    b = Move(1, Position(5, 0, 0), Position(10, 0, 0))
    assert not compatible_2d(a, b)


def test_clearance_rejects_close_parallel_moves():
    a = Move(0, Position(0, 0, 0), Position(10, 0, 0), clearance_um=2.0)
    b = Move(1, Position(3, 0, 0), Position(13, 0, 0), clearance_um=2.0)
    assert not compatible_2d(a, b)


def test_clearance_accepts_sufficiently_separated_parallel_moves():
    a = Move(0, Position(0, 0, 0), Position(10, 0, 0), clearance_um=1.0)
    b = Move(1, Position(3, 0, 0), Position(13, 0, 0), clearance_um=1.0)
    assert compatible_2d(a, b)


def test_clearance_zero_matches_bare_point_behaviour():
    a = Move(0, Position(0, 0, 0), Position(10, 0, 0))
    b = Move(1, Position(3, 0, 0), Position(13, 0, 0))
    assert compatible_2d(a, b)


def test_legalize_frames_separates_a_genuine_conflict():
    moves = [
        Move(0, Position(0, 0, 0), Position(10, 0, 0)),
        Move(1, Position(5, 0, 0), Position(2, 0, 0)),  # crosses move 0
        Move(2, Position(0, 10, 0), Position(0, 20, 0)),  # independent
    ]
    frames = legalise_frames(moves)
    frame_of = {m.qubit: i for i, frame in enumerate(frames) for m in frame}
    assert frame_of[0] != frame_of[1]
    for frame in frames:
        for i in range(len(frame)):
            for j in range(i + 1, len(frame)):
                assert compatible_2d(frame[i], frame[j])


def test_legalize_frames_every_move_scheduled_exactly_once():
    moves = [
        Move(0, Position(0, 0, 0), Position(10, 0, 0)),
        Move(1, Position(5, 0, 0), Position(2, 0, 0)),
        Move(2, Position(0, 10, 0), Position(0, 20, 0)),
        Move(3, Position(-5, -5, 0), Position(5, 5, 0)),
    ]
    frames = legalise_frames(moves)
    all_qubits = [m.qubit for frame in frames for m in frame]
    assert sorted(all_qubits) == [0, 1, 2, 3]


def test_independent_moves_in_separate_regions_share_one_frame():
    moves = [
        Move(0, Position(0, 0, 0), Position(10, 0, 0), clearance_um=2.0),
        Move(1, Position(0, 100, 0), Position(10, 100, 0), clearance_um=2.0),
        Move(2, Position(100, 0, 0), Position(110, 0, 0), clearance_um=2.0),
    ]
    frames = legalise_frames(moves)
    assert len(frames) == 1


def test_move_duration_matches_minimum_jerk_formula(model):
    move = Move(0, Position(0, 0, 0), Position(11, 0, 0))
    expected = (15 / 8) * (11.0 / model.v_xy_um_per_us)
    assert move_duration_us(move, model) == pytest.approx(expected)


def test_moves_between_only_includes_relocated_qubits(model, device):
    circuit = Circuit.from_stages(3, [[two_qubit(0, 1)]])
    before = piqasso_placement(circuit, model, device, distance=3, seed=0)
    after = piqasso_placement(circuit, model, device, distance=3, seed=1)
    moves = moves_between(before, after)
    before_anchor = {p.qubit: p.anchor for p in before.patches}
    after_anchor = {p.qubit: p.anchor for p in after.patches}
    for m in moves:
        assert before_anchor[m.qubit] != after_anchor[m.qubit]
        assert m.clearance_um > 0

def test_legalize_frames_raises_on_unresolvable_cycle():
    a = Position(0, 0, 0)
    b = Position(10, 0, 0)
    c = Position(20, 0, 0)
    moves = [
        Move(0, a, b),
        Move(1, b, c),
        Move(2, c, a),
    ]
    with pytest.raises(ValueError, match="cycle"):
        legalise_frames(moves)


def test_legalize_frames_resolves_a_chain_by_ordering_frames():
    a = Position(0, 0, 0)
    b = Position(10, 0, 0)
    c = Position(20, 0, 0)
    moves = [
        Move(0, a, b),  # q0 moves onto q1's current site
        Move(1, b, c),  # q1 moves onto an empty site
    ]
    frames = legalise_frames(moves)
    frame_of = {m.qubit: i for i, frame in enumerate(frames) for m in frame}
    assert frame_of[1] < frame_of[0]  # q1 must vacate before q0 can land on b

def test_shared_start_row_gets_parked_into_one_batch():
    a = Move(qubit=0, source=Position(0, 0), target=Position(0, 50))
    b = Move(qubit=1, source=Position(50, 0), target=Position(50, -50))
    assert not compatible_2d(a, b)

    frames = legalise_frames([a, b])

    assert len(frames) == 2
    park_frame, main_frame = frames
    assert len(park_frame) == 1
    assert len(main_frame) == 2
    assert {m.target for m in main_frame} == {a.target, b.target}


def test_parking_clears_clearance_margin():
    clearance = 18.75
    a = Move(qubit=0, source=Position(0, 0), target=Position(0, 100), clearance_um=clearance)
    b = Move(qubit=1, source=Position(80, 0), target=Position(80, -100), clearance_um=clearance)

    frames = legalise_frames([a, b])
    park = frames[0][0]
    other = b if park.qubit == a.qubit else a
    d = math.hypot(park.target.x - other.source.x, park.target.y - other.source.y)
    assert d >= park.clearance_um + other.clearance_um - 1e-9


def test_genuine_crossing_still_serialises():
    a = Move(qubit=0, source=Position(0, 0), target=Position(100, 0))
    b = Move(qubit=1, source=Position(100, 10), target=Position(0, 10))
    assert not compatible_2d(a, b)

    frames = legalise_frames([a, b])

    assert len(frames) == 2
    assert all(len(f) == 1 for f in frames)


def test_independent_moves_unaffected():
    a = Move(qubit=0, source=Position(0, 0), target=Position(0, 10))
    b = Move(qubit=1, source=Position(200, 0), target=Position(200, 10))
    c = Move(qubit=2, source=Position(400, 0), target=Position(400, 10))

    frames = legalise_frames([a, b, c])

    assert len(frames) == 1
    assert len(frames[0]) == 3


def test_pairwise_compatible_moves_still_batch_with_no_stationary_bystander():
    # a and b are pairwise-compatible (different x-tones, disjoint y-ranges,
    # no crossing/shared-tone conflict) - with nothing stationary to protect,
    # they should still share one frame.
    a = Move(qubit=0, source=Position(0, 0), target=Position(0, 100))
    b = Move(qubit=1, source=Position(50, 200), target=Position(50, 300))
    assert compatible_2d(a, b)

    frames = legalise_frames([a, b])
    assert len(frames) == 1
    assert len(frames[0]) == 2


def test_stationary_bystander_at_a_tone_intersection_forces_a_split():
    # Same a/b as above - pairwise compatible - but their SOURCE tones
    # {x=0,50} x {y=0,200} include two combinations (0,200) and (50,0)
    # neither move actually uses. A stationary patch sitting at one of
    # those spurious intersections would get swept up by the AOD if a
    # and b fired in the same pulse, even though compatible_2d sees no
    # pairwise conflict at all.
    a = Move(qubit=0, source=Position(0, 0), target=Position(0, 100))
    b = Move(qubit=1, source=Position(50, 200), target=Position(50, 300))
    bystander = Position(0, 200)

    frames = legalise_frames([a, b], stationary=[bystander])

    frame_of = {m.qubit: i for i, frame in enumerate(frames) for m in frame}
    assert frame_of[0] != frame_of[1]


def test_stationary_bystander_off_the_tone_grid_does_not_affect_batching():
    a = Move(qubit=0, source=Position(0, 0), target=Position(0, 100))
    b = Move(qubit=1, source=Position(50, 200), target=Position(50, 300))
    far_away = Position(9000, 9000)

    frames = legalise_frames([a, b], stationary=[far_away])
    assert len(frames) == 1
    assert len(frames[0]) == 2


def test_stationary_bystander_at_an_intended_source_is_not_a_false_positive():
    # (0, 0) is move a's own source, not a spurious combination - a
    # bystander can't physically sit exactly where a is already a real
    # mover, but this pins down that _aod violations only flags the
    # extra combinations, not the intended sources themselves.
    a = Move(qubit=0, source=Position(0, 0), target=Position(0, 100))
    b = Move(qubit=1, source=Position(50, 200), target=Position(50, 300))

    frames = legalise_frames([a, b], stationary=[Position(0, 0)])
    assert len(frames) == 1
    assert len(frames[0]) == 2