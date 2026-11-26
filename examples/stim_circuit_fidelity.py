"""
Computes end-to-end circuit fidelity for the same Clifford+T circuit
families used in examples/qldpc_scaling_experiment.py's spacetime-volume
comparison (so the two can be plotted side by side against circuit
size), at a dense grid of deployment-range p_ph (1e-4 to 4e-3), using
per-gadget logical error rates from the power-law extrapolation in
examples/subthreshold_scaling.py (nasim.fidelity's Stim-calibrated
gadgets: merge/split, surface-code memory, qLDPC memory).

Why read extrapolated rates instead of calibrating live at this p_ph?
Direct Monte Carlo there is intractable -- that's the whole reason
examples/subthreshold_scaling.py exists (fit a power law at measurable
sub-threshold points, extrapolate down/interpolate up to 4e-3). Run that
script first; this one reads its output
(examples/subthreshold_scaling_results.csv) rather than re-calibrating.

Circuits: every (family, label) in CIRCUITS below -- the cirq-ft and
QASMBench Clifford+T families from qldpc_scaling_experiment.py (qrom,
mcx, hb_sel, adder, cc, ising, wstate), all sizes. All real enough to
have a nonzero Nt (magic-state-consumption) count so the merge/split-based
Nt path gets exercised, not just Ncnot -- unlike that script's ghz/bv
families, which are T-free, so those are deliberately NOT included here.
Every fixture needed is already committed (no generate_large_circuits.py
step required).

This is a lot more compute than the original 3-circuit sweep (23
circuits x 15 p_ph points = 345 schedule builds, some on circuits with
100+ qubits), so results are written incrementally (one row at a time,
flushed immediately) and any (p_ph, circuit) pair already present in
RESULTS_CSV is skipped on a rerun -- safe to interrupt and resume, and
safe to extend CIRCUITS/P_PHYS_POINTS later without repeating work.

Run from the repo root, in the MAIN venv, AFTER subthreshold_scaling.py:

    python examples/subthreshold_scaling.py
    python examples/stim_circuit_fidelity.py
"""

import csv
import math
import random
from pathlib import Path

from nasim.device import Device
from nasim.fidelity import circuit_fidelity_from_rates
from nasim.fidelity.circuit_fidelity import SURFACE_DIST
from nasim.logical_error_model import logical_error_rate
from nasim.model import Model
from nasim.placement import patch_separation_um
from nasim.qasm import load_clifford_t_circuit
from nasim.qldpc import BB_144_12_12
from nasim.resource import Factory
from nasim.schedule.qldpc_memory import MemoryBlock
from nasim.schedule.qldpc_simulate import simulate_qldpc_circuit
from nasim.geometry import Position

PRESET = Path("src/nasim/config/presets/2d.yaml")

# (label, fixture_dir) -- matches qldpc_scaling_experiment.py's
# cirqft_t_families + qasmbench_t_families exactly (minus its T-free
# ghz/bv families -- see module docstring).
CIRCUITS = [
    *[(label, "tests/fixtures/cirq_ft_tgate") for label in
      ["qrom", "qrom_s1", "qrom_s2", "qrom_s3"]],
    *[(label, "tests/fixtures/cirq_ft_tgate") for label in
      ["mcx", "mcx_s1", "mcx_s2", "mcx_s3"]],
    *[(label, "tests/fixtures/cirq_ft_tgate") for label in
      ["hb_sel", "hb_sel_s1", "hb_sel_s2", "hb_sel_s3"]],
    *[(label, "tests/fixtures/qasmbench_tgate") for label in
      ["adder_s1", "adder_s2", "adder_s3"]],
    *[(label, "tests/fixtures/qasmbench_tgate") for label in
      ["cc_s1", "cc_s2", "cc_s3"]],
    *[(label, "tests/fixtures/qasmbench_tgate") for label in
      ["ising_s1", "ising_s2", "ising_s3"]],
    *[(label, "tests/fixtures/qasmbench_tgate") for label in
      ["wstate_s1", "wstate_s2"]],
]

# must match subthreshold_scaling.py's TARGET_POINTS
P_PHYS_POINTS = [
    1e-4, 1.5e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 8e-4,
    1e-3, 1.5e-3, 2e-3, 2.5e-3, 3e-3, 3.5e-3, 4e-3,
]

QLDPC_CODE_DISTANCE = 12
SURFACE_DISTANCE = 13
NUM_FACTORIES = 2

EXTRAPOLATED_RATES_CSV = Path("examples/subthreshold_scaling_results.csv")
RESULTS_CSV = Path("examples/tier2_fidelity_sanity_check_results.csv")


def load_extrapolated_rates(csv_path: Path) -> dict[tuple[str, float], float]:
    """gadget, p_phys -> predicted_per_round_rate, straight from subthreshold_scaling.py's output."""
    if not csv_path.exists():
        raise SystemExit(f"{csv_path} not found -- run examples/subthreshold_scaling.py first")

    rates: dict[tuple[str, float], float] = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rates[(row["gadget"], float(row["p_phys"]))] = float(row["predicted_per_round_rate"])
    return rates


def _idle_rounds_by_family(schedule, blocks: dict | None):
    qldpc_ids = set(blocks) if blocks else set()
    per_qubit_idle_us: dict[int, float] = {}
    for (qubit, _), t in schedule.idle_us.items():
        per_qubit_idle_us[qubit] = max(per_qubit_idle_us.get(qubit, 0.0), t)

    qec_us = schedule.model.qec_round_duration_us
    surface_idle_us = sum(t for q, t in per_qubit_idle_us.items() if q not in qldpc_ids)
    qldpc_idle_us = sum(t for q, t in per_qubit_idle_us.items() if q in qldpc_ids)
    n_surface = (surface_idle_us / qec_us) / schedule.model.logical_se_interval
    n_qldpc = (qldpc_idle_us / qec_us) / schedule.model.logical_se_interval
    return n_surface, n_qldpc


def log10_circuit_fidelity_from_rates(schedule, *, p_ph: float, blocks: dict | None,
                                      merge_split_rate: float, surface_rate: float, qldpc_rate: float,
                                      cultivation_t_state_infidelity: float) -> float:
    """
    Log-space mirror of circuit_fidelity_from_rates -- real circuits
    underflow double-precision float to exactly 0.0 for a large enough
    op count, well before the log is anywhere near interesting, so the
    raw fidelity alone isn't reliable for the larger circuits/higher p_ph
    points in this sweep.
    """
    log_f = schedule.Ncnot * math.log10(1.0 - merge_split_rate)
    log_f += schedule.Nt * math.log10(1.0 - merge_split_rate)
    log_f += schedule.Nt * math.log10(1.0 - cultivation_t_state_infidelity)

    n_1q = sum(len(op.detail["qubits"]) for op in schedule.ops if op.kind == "1q_stage")
    log_f += n_1q * math.log10(1.0 - logical_error_rate("H", p_ph, SURFACE_DIST))

    n_s = sum(1 for op in schedule.ops if op.kind == "correction_gate")
    log_f += n_s * math.log10(1.0 - logical_error_rate("S", p_ph, SURFACE_DIST))

    n_surface_idle_rounds, n_qldpc_idle_rounds = _idle_rounds_by_family(schedule, blocks)
    log_f += n_surface_idle_rounds * math.log10(1.0 - surface_rate)
    log_f += n_qldpc_idle_rounds * math.log10(1.0 - qldpc_rate)

    return log_f


def build_schedule(label: str, fixture_dir: str, model: Model):
    n, gates = load_clifford_t_circuit(str(Path(fixture_dir) / f"{label}.qasm"))
    from nasim.circuit import Circuit
    circuit = Circuit.from_gates(n, gates).with_separated_magic_stages()

    n_blocks = math.ceil(circuit.num_qubits / BB_144_12_12.k)
    capacity = 2 * n_blocks
    blocks = {
        -(i + 1): MemoryBlock(block_id=-(i + 1), code=BB_144_12_12, anchor=Position(i * 500.0, 0.0))
        for i in range(n_blocks)
    }
    factories = [
        Factory(resource_id=-(1000 + i), model=model, rng=random.Random(i), distance=SURFACE_DISTANCE)
        for i in range(NUM_FACTORIES)
    ]

    pitch = patch_separation_um(model, distance=SURFACE_DISTANCE)
    side = max(6, math.ceil(math.sqrt(circuit.num_qubits)) + 3)
    device = Device.zoned(
        model=model,
        storage_rows=side, storage_cols=side, storage_pitch_um=pitch,
        entanglement_rows=side, entanglement_cols=side, entanglement_pitch=pitch,
        gap_um=2 * pitch,
    )

    schedule, current, assignment, occupancies, residency = simulate_qldpc_circuit(
        circuit, model, device, blocks, factories,
        qldpc_code_distance=QLDPC_CODE_DISTANCE, surface_distance=SURFACE_DISTANCE, compute_capacity=capacity,
        track_idle=True,
    )
    return schedule, blocks


FIELDNAMES = ["p_ph", "circuit", "Ncnot", "Nt", "fidelity", "log10_fidelity"]


def load_completed(csv_path: Path) -> set[tuple[float, str]]:
    """(p_ph, circuit) pairs already present -- skip these on a rerun."""
    if not csv_path.exists():
        return set()
    with open(csv_path) as f:
        return {(float(row["p_ph"]), row["circuit"]) for row in csv.DictReader(f)}


def main() -> None:
    model = Model.from_yaml(PRESET)
    extrapolated_rates = load_extrapolated_rates(EXTRAPOLATED_RATES_CSV)
    completed = load_completed(RESULTS_CSV)
    if completed:
        print(f"{len(completed)} (p_ph, circuit) points already in {RESULTS_CSV} -- skipping those")

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not RESULTS_CSV.exists()
    n_written = 0

    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
            f.flush()

        for label, fixture_dir in CIRCUITS:
            # build_schedule (full placement/routing/qLDPC-scheduling
            # simulation) does not depend on p_ph at all -- only the
            # final circuit_fidelity_from_rates arithmetic does. Building
            # it once per circuit instead of once per (p_ph, circuit)
            # cuts the expensive part of this script's work by roughly
            # len(P_PHYS_POINTS)x.
            pending_p_ph = [p for p in P_PHYS_POINTS if (p, label) not in completed]
            if not pending_p_ph:
                continue

            print(f"circuit={label} ({len(pending_p_ph)}/{len(P_PHYS_POINTS)} p_ph points pending) ...")
            schedule, blocks = build_schedule(label, fixture_dir, model)
            print(f"  Ncnot={schedule.Ncnot} Nt={schedule.Nt}")

            for p_ph in pending_p_ph:
                try:
                    merge_split_rate = extrapolated_rates[("merge_split", p_ph)]
                    surface_rate = extrapolated_rates[("surface_memory", p_ph)]
                    qldpc_rate = extrapolated_rates[("qldpc_memory", p_ph)]
                except KeyError:
                    print(f"  p_ph={p_ph}: no extrapolated rate for one or more gadgets in {EXTRAPOLATED_RATES_CSV} -- "
                          f"add it to subthreshold_scaling.py's TARGET_POINTS and rerun that script first. Skipping.")
                    continue

                fidelity = circuit_fidelity_from_rates(
                    schedule, p_ph=p_ph, merge_split_rate=merge_split_rate, surface_rate=surface_rate,
                    qldpc_rate=qldpc_rate, cultivation_t_state_infidelity=model.cultivation_t_state_infidelity_target,
                    blocks=blocks,
                )
                # raw fidelity underflows to exactly 0.0 for a large enough
                # circuit at this p_ph -- log10 is the reliable column.
                log10_fidelity = log10_circuit_fidelity_from_rates(
                    schedule, p_ph=p_ph, blocks=blocks, merge_split_rate=merge_split_rate, surface_rate=surface_rate,
                    qldpc_rate=qldpc_rate, cultivation_t_state_infidelity=model.cultivation_t_state_infidelity_target,
                )

                print(f"  p_ph={p_ph}: log10(fidelity)={log10_fidelity:.3f} raw fidelity={fidelity:.6f}")
                writer.writerow({
                    "p_ph": p_ph, "circuit": label, "Ncnot": schedule.Ncnot, "Nt": schedule.Nt,
                    "fidelity": fidelity, "log10_fidelity": log10_fidelity,
                })
                f.flush()
                n_written += 1

    print(f"\nwrote {n_written} new rows to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
