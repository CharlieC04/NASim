from pathlib import Path

import pytest

from nasim.model import Model
from nasim.schedule import circuit_fidelity
from nasim.schedule.core import Schedule, ScheduleOp

PRESET = Path(__file__).resolve().parents[1] / "src" / "nasim" / "config" / "presets" / "2d_nozone.yaml"

# FT-Weave's own published d=7 row (UCLA-VAST/FT-Weave, MIT licensed,
# output/evaluation/fidelity/t_cultivation_fidelity_results.csv). This is
# the first end-to-end check of circuit_fidelity's full composition
# against a real published number - earlier validation only checked the
# per-gate logical_error_rate formula in isolation (test_logical_error_model.py).
N_CNOT = 64
N_1Q = 2624
N_T = 2528
N_SE_Q = 13883  # idle exposure count during syndrome extraction

FT_WEAVE_FIDELITY_CNOT = 0.9998080181428767
FT_WEAVE_FIDELITY_H = 0.9989509504276834
FT_WEAVE_FIDELITY_OF_T_GATE = 0.9999747203192831
FT_WEAVE_FIDELITY_IDLE = 0.9972272509115823

# The "achievable subset" of their published fidelity_total: CNOT x H x T
# x idle only - excludes their fidelity_of_rz_teleportaion/fidelity_synthesis
# terms, which are specific to their Rz-rotation-synthesis circuit
# structure and have no equivalent in our (non-synthesis) circuit model.
TARGET_FIDELITY = (
    FT_WEAVE_FIDELITY_CNOT * FT_WEAVE_FIDELITY_H * FT_WEAVE_FIDELITY_OF_T_GATE * FT_WEAVE_FIDELITY_IDLE
)


def test_circuit_fidelity_reproduces_ft_weave_d7_published_number():
    model = Model.from_yaml(PRESET)
    qec_us = model.qec_round_duration_us

    atom = (0, (0, 0))
    schedule = Schedule(model=model, all_atoms={atom})
    schedule.Ncnot = N_CNOT
    schedule.Nt = N_T
    schedule.idle_us[atom] = N_SE_Q * qec_us
    schedule.ops = [
        ScheduleOp(kind="1q_stage", start_us=0.0, duration_us=0.0, targets=(), detail={"qubits": [0]})
        for _ in range(N_1Q)
    ]

    f = circuit_fidelity(schedule, d=7)

    assert f == pytest.approx(TARGET_FIDELITY)
