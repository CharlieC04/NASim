import random
from pathlib import Path

import pytest

from nasim.circuit import Circuit, one_qubit, two_qubit, magic_consume
from nasim.device import Device
from nasim.model import Model
from nasim.placement import zap_placement
from nasim.resource import Factory
from nasim.schedule import circuit_fidelity, simulate_circuit

PRESET = Path(__file__).resolve().parents[1] / "src" / "nasim" / "config" / "presets" / "2d_nozone.yaml"
DISTANCE = 7
NUM_QUBITS = 4
NUM_FACTORIES = 2
ROUNDS = 4


@pytest.fixture
def model() -> Model:
    return Model.from_yaml(PRESET)


def _circuit() -> Circuit:
    stages = []
    for r in range(ROUNDS):
        stages.append([one_qubit(q) for q in range(NUM_QUBITS)])
        stages.append([magic_consume(q) for q in range(NUM_QUBITS)])
        pair = (r % (NUM_QUBITS - 1), r % (NUM_QUBITS - 1) + 1)
        stages.append([two_qubit(*pair)])
    return Circuit.from_stages(NUM_QUBITS, stages)


def test_device_from_single_zone_preset_has_only_an_entanglement_zone(model):
    device = Device.from_yaml(PRESET, model=model)
    assert [z.kind for z in device.zones] == ["entanglement"]


def test_zap_placement_places_qubits_and_factories_in_the_one_zone(model):
    device = Device.from_yaml(PRESET, model=model)
    circuit = _circuit()
    placement = zap_placement(circuit, model, device, distance=DISTANCE)

    ent_zone = device.zones[0]
    assert all(ent_zone.contains(p.anchor) for p in placement.patches)


def test_end_to_end_pipeline_on_single_zone_device_produces_a_valid_fidelity(model):
    device = Device.from_yaml(PRESET, model=model)
    circuit = _circuit()
    placement = zap_placement(circuit, model, device, distance=DISTANCE)
    factories = [
        Factory(resource_id=-(i + 1), model=model, rng=random.Random(i), distance=DISTANCE, num_subfactories=2)
        for i in range(NUM_FACTORIES)
    ]

    schedule, final = simulate_circuit(circuit, placement, model, device=device, factories=factories)
    f = circuit_fidelity(schedule, d=DISTANCE)

    assert 0.0 < f < 1.0
    assert schedule.Ncnot == ROUNDS
    assert schedule.Nt == ROUNDS * NUM_QUBITS

    ent_zone = device.zones[0]
    assert all(ent_zone.contains(p.anchor) for p in final.patches), (
        "with no storage zone, every qubit and factory must stay resident in the one zone"
    )
