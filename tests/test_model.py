from pathlib import Path

import pytest

from nasim.model import Model

PRESET = Path(__file__).resolve().parents[1] / "src" / "nasim" / "config" / "presets" / "2d.yaml"


@pytest.fixture
def model() -> Model:
    return Model.from_yaml(PRESET)


def test_loads_harvard_2d_identity(model):
    assert model.name == "harvard_2d"
    assert model.dimensions == "2D"
    assert model.layers == 1
    assert model.layer_spacing_um == 0.0


def test_transport_params(model):
    assert model.v_xy_um_per_us == 0.55
    assert model.v_loss_um_per_us == 1.0
    assert model.handover_time_us == 15.0
    assert model.handover_fidelity == 0.999
    assert model.move_profile == "min_jerk"


def test_gate_params(model):
    assert model.cz_fidelity == 0.995
    assert model.cz_duration_us == 0.36
    assert model.single_qubit_fidelity == 0.9975
    assert model.single_qubit_duration_us == 0.625
    assert model.carrier_fidelity == 0.9997


def test_readout_and_coherence(model):
    assert model.readout_fidelity == 0.998
    assert model.readout_time_us == 500.0
    assert model.t2_s == 12.6


def test_interaction_and_geometry(model):
    assert model.blockade_radius_um == 5.0
    assert model.gate_pair_dist_um == 2.5
