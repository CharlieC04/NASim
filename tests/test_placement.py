import math
from pathlib import Path

import pytest

from nasim.circuit import Circuit, one_qubit, two_qubit
from nasim.device import Device
from nasim.model import Model
from nasim.placement import patch_separation_um, piqasso_placement, zap_placement

PRESET = Path(__file__).resolve().parents[1] / "src" / "nasim" / "config" / "presets" / "2d.yaml"


@pytest.fixture
def model() -> Model:
    return Model.from_yaml(PRESET)


@pytest.fixture
def device(model) -> Device:
    return Device.uniform(model=model, rows=12, cols=12, pitch_um=patch_separation_um(model, distance=3))


@pytest.fixture
def zoned_device(model) -> Device:
    pitch = patch_separation_um(model, distance=3)
    return Device.zoned(
        model=model,
        storage_rows=4, storage_cols=4, storage_pitch_um=pitch,
        entanglement_rows=2, entanglement_cols=2, entanglement_pitch=pitch,
        gap_um=pitch,
    )


def test_places_one_patch_per_qubit(model, device):
    circuit = Circuit.from_stages(4, [[two_qubit(0, 1)], [two_qubit(2, 3)]])
    placement = piqasso_placement(circuit, model, device, distance=3, seed=0)
    assert len(placement.patches) == 4
    assert {p.qubit for p in placement.patches} == {0, 1, 2, 3}


def test_patches_do_not_overlap(model, device):
    circuit = Circuit.from_stages(6, [[two_qubit(i, i + 1)] for i in range(5)])
    placement = piqasso_placement(circuit, model, device, distance=3, seed=2, iterations=300)
    all_positions = [(pos.x, pos.y) for _, _, pos in placement.atoms()]
    assert len(all_positions) == len(set(all_positions))


def test_interacting_qubits_end_up_closer_than_non_interacting(model, device):
    circuit = Circuit.from_stages(4, [[two_qubit(0, 1)]])
    placement = piqasso_placement(circuit, model, device, distance=3, seed=3, iterations=300)
    anchors = {p.qubit: p.anchor for p in placement.patches}

    d01 = math.hypot(anchors[0].x - anchors[1].x, anchors[0].y - anchors[1].y)
    d02 = math.hypot(anchors[0].x - anchors[2].x, anchors[0].y - anchors[2].y)
    assert d01 < d02


def test_qubits_with_no_interactions_stay_within_the_bounding_region(model, device):
    """Regression test: qubits with zero interaction weight (pure repulsion,
    nothing pulling them back) must not be able to drift to extreme
    distances, since Piqasso clips every position to a bounding region
    each iteration (\u00a7IV-A)."""
    circuit = Circuit.from_stages(4, [[two_qubit(0, 1)]])  # qubits 2, 3 are isolated
    placement = piqasso_placement(circuit, model, device, distance=3, seed=3, iterations=300)
    anchors = {p.qubit: p.anchor for p in placement.patches}

    d_star_estimate = max(
        abs(anchors[i].x - anchors[j].x) + abs(anchors[i].y - anchors[j].y)
        for i in anchors for j in anchors if i != j
    )
    # loose bound: nothing should be placed many hundreds of patch-widths away
    assert d_star_estimate < 500


def test_zap_placement_only_uses_storage_zone_sites(model, zoned_device):
    circuit = Circuit.from_stages(4, [[two_qubit(0, 1)], [two_qubit(2, 3)]])
    placement = zap_placement(circuit, model, zoned_device, distance=3)

    storage_zone = next(z for z in zoned_device.zones if z.kind == "storage")
    for p in placement.patches:
        assert storage_zone.contains(p.anchor)


def test_zap_placement_is_deterministic(model, zoned_device):
    circuit = Circuit.from_stages(4, [[two_qubit(0, 1)], [two_qubit(2, 3)]])
    a = zap_placement(circuit, model, zoned_device, distance=3)
    b = zap_placement(circuit, model, zoned_device, distance=3)

    anchors_a = {p.qubit: p.anchor for p in a.patches}
    anchors_b = {p.qubit: p.anchor for p in b.patches}
    assert anchors_a == anchors_b


def test_zap_placement_prioritizes_earlier_qubits_with_closer_sites(model, zoned_device):
    # qubit 0 is needed in stage 0 (weight 1.0), qubit 1 only in stage 1
    # (weight 0.5) - qubit 0 should be placed first and therefore get the
    # storage site closer to the entanglement zone.
    circuit = Circuit.from_stages(2, [[one_qubit(0)], [one_qubit(1)]])
    placement = zap_placement(circuit, model, zoned_device, distance=3)
    anchors = {p.qubit: p.anchor for p in placement.patches}

    ent_zone = next(z for z in zoned_device.zones if z.kind == "entanglement")
    ent_sites = ent_zone.sites

    def d_min(pos):
        return min(math.hypot(pos.x - e.x, pos.y - e.y) for e in ent_sites)

    assert d_min(anchors[0]) <= d_min(anchors[1])


def test_zap_placement_does_not_double_occupy_a_site(model, zoned_device):
    circuit = Circuit.from_stages(6, [[two_qubit(i, i + 1)] for i in range(5)])
    placement = zap_placement(circuit, model, zoned_device, distance=3)
    anchors = [p.anchor for p in placement.patches]
    assert len(anchors) == len(set(anchors))


def test_zap_placement_rejects_a_device_without_zones(model, device):
    circuit = Circuit.from_stages(2, [[two_qubit(0, 1)]])
    with pytest.raises(ValueError):
        zap_placement(circuit, model, device, distance=3)


def test_zap_placement_rejects_circuit_too_big_for_storage_zone(model, zoned_device):
    circuit = Circuit.from_stages(20, [[one_qubit(q) for q in range(20)]])
    with pytest.raises(ValueError):
        zap_placement(circuit, model, zoned_device, distance=3)
