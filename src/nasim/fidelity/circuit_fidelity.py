from __future__ import annotations

from pathlib import Path

from nasim.fidelity.cache import cached_calibrate
from nasim.fidelity.calibrate import calibrate_gen_bicycle_code, calibrate_merge_split, calibrate_surface_code
from nasim.logical_error_model import logical_fidelity
from nasim.qldpc import BB_144_12_12
from nasim.schedule.core import Schedule

SURFACE_DIST = 13
SURFACE_K = (SURFACE_DIST - 1) // 2
QLDPC_ROUNDS = 12

CACHE_DIR = Path("examples/calibration_cache")
MERGE_SPLIT_CACHE = CACHE_DIR / "merge_split.csv"
SURFACE_CACHE = CACHE_DIR / "surface_memory.csv"
QLDPC_CACHE = CACHE_DIR / "qldpc_memory.csv"

def circuit_fidelity(schedule: Schedule, *, p_ph: float, blocks: dict | None = None,
                     qldpc_code=BB_144_12_12, cultivation_t_state_infidelity: float, shots: int = 4000) -> float:

    merge_result = cached_calibrate(
        MERGE_SPLIT_CACHE, {"gadget": "merge_split", "k": SURFACE_K, "basis": "Z", "shots": shots},
        p_phys=p_ph,
        compute=lambda: calibrate_merge_split(k=SURFACE_K, p_phys=p_ph, shots=shots),
    )
    p_merge_fail = merge_result.logical_error_rate

    f = (1.0 - p_merge_fail) ** schedule.Ncnot
    f *= (1 - p_merge_fail) ** schedule.Nt
    f *= (1.0 - cultivation_t_state_infidelity) ** schedule.Nt

    n_1q = sum(len(op.detail["qubits"]) for op in schedule.ops if op.kind == "1q_stage")
    f *= logical_fidelity("H", p_ph, SURFACE_DIST) ** n_1q

    n_s = sum(1 for op in schedule.ops if op.kind == "correction_gate")
    f *= logical_fidelity("S", p_ph, SURFACE_DIST) ** n_s

    qldpc_ids = set(blocks) if blocks else set()
    per_qubit_idle_us: dict[int, float] = {}
    for (qubit, _), t in schedule.idle_us.items():
        per_qubit_idle_us[qubit] = max(per_qubit_idle_us.get(qubit, 0.0), t)

    qec_us = schedule.model.qec_round_duration_us
    surface_idle_us = sum(t for q, t in per_qubit_idle_us.items() if q not in qldpc_ids)
    qldpc_idle_us = sum(t for q, t in per_qubit_idle_us.items() if q in qldpc_ids)

    n_surface_idle_rounds = (surface_idle_us / qec_us) / schedule.model.logical_se_interval
    n_qldpc_idle_rounds = (qldpc_idle_us / qec_us) / schedule.model.logical_se_interval

    if n_surface_idle_rounds > 0:
        surface_cal = cached_calibrate(
            SURFACE_CACHE, {"gadget": "surface_memory", "distance": SURFACE_DIST, "rounds": QLDPC_ROUNDS, "shots": shots},
            p_phys=p_ph,
            compute=lambda: calibrate_surface_code(distance=SURFACE_DIST, p_phys=p_ph, rounds=QLDPC_ROUNDS, shots=shots),
        )
        f *= (1.0 - surface_cal.per_round_logical_error_rate) ** n_surface_idle_rounds

    if n_qldpc_idle_rounds > 0:
        qldpc_cal = cached_calibrate(
            QLDPC_CACHE, {"gadget": "qldpc_memory", "code": qldpc_code.name, "rounds": QLDPC_ROUNDS, "fast": True, "shots": shots},
            p_phys=p_ph,
            compute=lambda: calibrate_gen_bicycle_code(qldpc_code, p_phys=p_ph, rounds=QLDPC_ROUNDS, shots=shots, fast=True),
        )
        f *= (1.0 - qldpc_cal.per_round_logical_error_rate) ** n_qldpc_idle_rounds

    return f


def circuit_fidelity_from_rates(schedule: Schedule, *, p_ph: float, merge_split_rate: float, surface_rate: float,
                                qldpc_rate: float, cultivation_t_state_infidelity: float,
                                blocks: dict | None = None) -> float:

    """
    Same term structure as circuit_fidelity, but takes pre-computed
    per-round-equivalent rates instead of calibrating them live. For a
    deployment-range p_ph (e.g. 1e-4-5e-4), live calibration is
    intractable -- see examples/subthreshold_scaling.py's power-law fit
    and extrapolation. merge_split_rate/surface_rate/qldpc_rate here are
    exactly subthreshold_scaling_results.csv's predicted_per_round_rate
    column, read for the matching p_phys.

    merge_split_rate is used as a per-EVENT rate (not per-round -- a
    merge/split is one atomic gadget), matching calibrate_merge_split's
    own CalibrationResult.rounds=1 convention; surface_rate/qldpc_rate
    are genuinely per-round, matching QLDPC_ROUNDS-normalized calibration.

    p_ph is still needed here even though the Stim-calibrated terms don't
    use it directly -- the H/S single-qubit terms were never Stim-calibrated
    (same as in circuit_fidelity) and still come from the analytic FT-Weave
    formula, which needs p_ph explicitly.
    """

    f = (1.0 - merge_split_rate) ** schedule.Ncnot
    f *= (1.0 - merge_split_rate) ** schedule.Nt
    f *= (1.0 - cultivation_t_state_infidelity) ** schedule.Nt

    n_1q = sum(len(op.detail["qubits"]) for op in schedule.ops if op.kind == "1q_stage")
    f *= logical_fidelity("H", p_ph, SURFACE_DIST) ** n_1q

    n_s = sum(1 for op in schedule.ops if op.kind == "correction_gate")
    f *= logical_fidelity("S", p_ph, SURFACE_DIST) ** n_s

    qldpc_ids = set(blocks) if blocks else set()
    per_qubit_idle_us: dict[int, float] = {}
    for (qubit, _), t in schedule.idle_us.items():
        per_qubit_idle_us[qubit] = max(per_qubit_idle_us.get(qubit, 0.0), t)

    qec_us = schedule.model.qec_round_duration_us
    surface_idle_us = sum(t for q, t in per_qubit_idle_us.items() if q not in qldpc_ids)
    qldpc_idle_us = sum(t for q, t in per_qubit_idle_us.items() if q in qldpc_ids)

    n_surface_idle_rounds = (surface_idle_us / qec_us) / schedule.model.logical_se_interval
    n_qldpc_idle_rounds = (qldpc_idle_us / qec_us) / schedule.model.logical_se_interval

    f *= (1.0 - surface_rate) ** n_surface_idle_rounds
    f *= (1.0 - qldpc_rate) ** n_qldpc_idle_rounds

    return f
