"""
Circuit-size scaling experiment: for each circuit family (two synthetic,
T-free families generated programmatically, and three real Clifford+T
families loaded from tests/fixtures/cirq_ft_tgate/), runs both the
qLDPC-memory architecture and the surface-code-only baseline across a
range of sizes, and records the spacetime-volume speedup
(baseline_volume / qldpc_volume -- >1 means qLDPC wins) to a CSV.

Run from the repo root, in the MAIN venv (not the isolated cirq-ft one):

    python examples/qldpc_scaling_experiment.py

This can take a while for the larger (>=100 qubit) points -- consider
running it in the background and checking progress via the CSV, which
is written incrementally (one row per completed point, flushed
immediately) rather than all at once at the end.

Before running the "qrom"/"mcx"/"hb_sel" large points, generate them
first -- see tests/fixtures/cirq_ft_tgate/generate_large_circuits.py's
own instructions (needs the separate, isolated cirq-ft environment).
Points whose QASM file doesn't exist yet are skipped with a printed
warning, not a hard failure, so this script still runs the ghz/bv
points (and the three small qrom/mcx/hb_sel circuits already committed)
even before the large ones are generated.
"""

import csv
import math
import random
from pathlib import Path

from qiskit import QuantumCircuit, transpile

from nasim.circuit import Circuit, one_qubit, two_qubit
from nasim.device import Device
from nasim.geometry import Position
from nasim.model import Model
from nasim.placement import patch_separation_um, zap_placement
from nasim.qasm import load_clifford_t_circuit
from nasim.qldpc import BB_144_12_12
from nasim.resource import Factory
from nasim.schedule import simulate_circuit
from nasim.schedule.qldpc_memory import MemoryBlock
from nasim.schedule.qldpc_metrics import baseline_spacetime_volume_us, qldpc_spacetime_volume_us
from nasim.schedule.qldpc_simulate import simulate_qldpc_circuit

PRESET = Path("src/nasim/config/presets/2d.yaml")
# d=3 was a placeholder from earlier correctness-only testing and gives a
# physically meaningless baseline (circuit_fidelity ~1e-58 on ising_s1 --
# essentially 0% success probability). The paper chooses code distance to
# hit a target per-operation logical error rate ("a conservative logical
# error rate estimate of 10^-9 for the physical error rate of 10^-3",
# Sec 8.3). At this model's own p_ph (~5e-4, derived from cz_fidelity),
# logical_error_rate("CNOT", p_ph, d) crosses below 1e-9 at d=13 (1.4e-10)
# -- d=11 is close (2.4e-9) but not quite under, and isn't in the
# cultivation_escape_cycles_by_distance table (7, 9, 13) anyway.
QLDPC_CODE_DISTANCE = 12
SURFACE_DISTANCE = 13
NUM_FACTORIES = 2
RESULTS_CSV = Path("examples/qldpc_scaling_results.csv")


# ---------- non-T circuit families (generated programmatically) ----------

def _transpile_to_gates(qc: QuantumCircuit):
    n_pre = qc.num_qubits
    opt_level = 3 if n_pre <= 24 else 2 if n_pre <= 64 else 1 if n_pre <= 128 else 0
    cz = transpile(qc, basis_gates=["cz", "id", "u2", "u1", "u3"], optimization_level=opt_level, seed_transpiler=0)
    gates = []
    for instr in cz.data:
        if instr.operation.num_qubits == 2:
            qa = cz.find_bit(instr.qubits[0]).index
            qb = cz.find_bit(instr.qubits[1]).index
            gates.append(two_qubit(qa, qb))
        elif instr.operation.name not in ("measure", "barrier"):
            q = cz.find_bit(instr.qubits[0]).index
            gates.append(one_qubit(q))
    return cz.num_qubits, gates


def ghz_circuit(n: int) -> Circuit:
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(1, n):
        qc.cx(0, i)
    num_qubits, gates = _transpile_to_gates(qc)
    return Circuit.from_gates(num_qubits, gates)


def bv_circuit(n: int) -> Circuit:
    qc = QuantumCircuit(n)
    qc.x(n - 1)
    qc.h(range(n))
    for i in range(0, n - 1, 2):
        qc.cx(i, n - 1)
    qc.h(range(n - 1))
    num_qubits, gates = _transpile_to_gates(qc)
    return Circuit.from_gates(num_qubits, gates)


# ---------- T-circuit families (loaded from committed/generated QASM) ----------

def load_tgate_circuit(label: str, fixture_dir: str = "tests/fixtures/cirq_ft_tgate") -> Circuit | None:
    path = Path(f"{fixture_dir}/{label}.qasm")
    if not path.exists():
        print(f"  [skip] {path} not found -- generate it first (see generate_large_circuits.py / "
              f"tests/fixtures/qasmbench_tgate/generate_*_circuits.py)")
        return None
    n, gates = load_clifford_t_circuit(str(path))
    return Circuit.from_gates(n, gates).with_separated_magic_stages()


# ---------- shared simulation helpers ----------

def device_for(model: Model, n: int) -> Device:
    # Sized proportionally to the circuit, not one fixed grid for every
    # size -- an oversized device inflates the baseline's movement cost
    # (zap_placement's greedy site-search scatters further when given
    # more room than it needs), which would confound the comparison.
    #
    # Entanglement zone is sized equal to storage (not halved): since
    # run_point's compute_capacity is fixed (2 per memory block, not tied
    # to the circuit -- see run_point), side**2 (storage zone size, always
    # > n by construction) comfortably covers it with room to spare.
    pitch = patch_separation_um(model, distance=SURFACE_DISTANCE)
    side = max(6, math.ceil(math.sqrt(n)) + 3)
    return Device.zoned(
        model=model,
        storage_rows=side, storage_cols=side, storage_pitch_um=pitch,
        entanglement_rows=side, entanglement_cols=side, entanglement_pitch=pitch,
        gap_um=2 * pitch,
    )


def run_point(circuit: Circuit, model: Model) -> tuple[float, float, int]:
    n = circuit.num_qubits
    device = device_for(model, n)

    # Fixed device parameter, not derived from the circuit: qSIEVE's own
    # device layout (Fig. 12) gives each LD/ST ancilla (one per memory
    # block) 2 dedicated surface codes, and Fig. 17's sensitivity sweep
    # runs from this "2 x (# Mem Block)" point upward -- i.e. it's their
    # evaluation default, not something that scales with circuit width. A
    # real compute zone can't resize itself per-circuit; the scheduler
    # (qldpc_simulate.py's _run_in_waves) delays whatever doesn't fit a
    # stage to a later round instead of requiring compute_capacity to
    # cover the circuit's widest stage.
    n_blocks = math.ceil(n / BB_144_12_12.k)
    capacity = 2 * n_blocks
    blocks = {
        -(i + 1): MemoryBlock(block_id=-(i + 1), code=BB_144_12_12, anchor=Position(i * 500.0, 0.0))
        for i in range(n_blocks)
    }
    factories = [
        Factory(resource_id=-(1000 + i), model=model, rng=random.Random(i), distance=SURFACE_DISTANCE)
        for i in range(NUM_FACTORIES)
    ]
    # track_idle=False: neither qldpc_spacetime_volume_us nor
    # baseline_spacetime_volume_us read schedule.idle_us (only
    # circuit_fidelity() does, which this script never calls) -- skipping
    # it avoids Schedule._advance's per-atom idle bookkeeping, which is
    # the dominant cost at larger device sizes.
    schedule, current, assignment, occupancies, residency = simulate_qldpc_circuit(
        circuit, model, device, blocks, factories,
        qldpc_code_distance=QLDPC_CODE_DISTANCE, surface_distance=SURFACE_DISTANCE, compute_capacity=capacity,
        track_idle=False,
    )
    q_vol = qldpc_spacetime_volume_us(schedule, blocks, SURFACE_DISTANCE)

    placement = zap_placement(circuit, model, device, distance=SURFACE_DISTANCE)
    factories2 = [
        Factory(resource_id=-(2000 + i), model=model, rng=random.Random(i), distance=SURFACE_DISTANCE)
        for i in range(NUM_FACTORIES)
    ]
    schedule2, final = simulate_circuit(circuit, placement, model, device=device, factories=factories2, track_idle=False)
    b_vol = baseline_spacetime_volume_us(schedule2, final, SURFACE_DISTANCE)

    return q_vol, b_vol, capacity


def main() -> None:
    model = Model.from_yaml(PRESET)

    # (family, is_t_family, points) -- points are either qubit counts
    # (non-T, built programmatically) or (label, qasm_label) pairs for
    # the T families (label is what's printed/plotted, qasm_label is the
    # fixture filename stem to load).
    non_t_families = [
        ("ghz", [20, 40, 60, 80, 100, 130, 160]),
        ("bv", [20, 40, 60, 80, 100, 130, 160]),
    ]
    # cirq-ft-sourced T families (tests/fixtures/cirq_ft_tgate/)
    cirqft_t_families = [
        ("qrom", "tests/fixtures/cirq_ft_tgate", ["qrom", "qrom_s1", "qrom_s2", "qrom_s3"]),
        ("mcx", "tests/fixtures/cirq_ft_tgate", ["mcx", "mcx_s1", "mcx_s2", "mcx_s3"]),
        ("hb_sel", "tests/fixtures/cirq_ft_tgate", ["hb_sel", "hb_sel_s1", "hb_sel_s2", "hb_sel_s3"]),
    ]

    # QASMBench-sourced families (tests/fixtures/qasmbench_tgate/) -- the
    # rest of qSIEVE's Table 5 benchmark suite: adder/cc exact Clifford+T,
    # ising/wstate via real Solovay-Kitaev synthesis (see that directory's
    # README.md for provenance). ghz/bv are also available there as the
    # paper's own exact circuit instances, in addition to the programmatic
    # non_t_families builders above.
    qasmbench_t_families = [
        ("adder", "tests/fixtures/qasmbench_tgate", ["adder_s1", "adder_s2", "adder_s3"]),
        ("cc", "tests/fixtures/qasmbench_tgate", ["cc_s1", "cc_s2", "cc_s3"]),
        ("ising", "tests/fixtures/qasmbench_tgate", ["ising_s1", "ising_s2", "ising_s3"]),
        ("wstate", "tests/fixtures/qasmbench_tgate", ["wstate_s1", "wstate_s2"]),
    ]

    write_header = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["family", "n", "capacity", "qldpc_volume_us", "baseline_volume_us", "speedup"])

        for family, sizes in non_t_families:
            builder = ghz_circuit if family == "ghz" else bv_circuit
            for n in sizes:
                print(f"=== {family} n={n} ===")
                circuit = builder(n)
                q_vol, b_vol, capacity = run_point(circuit, model)
                speedup = b_vol / q_vol
                print(f"  capacity={capacity} qldpc_vol={q_vol:.1f} baseline_vol={b_vol:.1f} speedup={speedup:.3f}")
                writer.writerow([family, n, capacity, q_vol, b_vol, speedup])
                f.flush()

        for family, fixture_dir, labels in cirqft_t_families + qasmbench_t_families:
            for label in labels:
                print(f"=== {family} ({label}) ===")
                circuit = load_tgate_circuit(label, fixture_dir)
                if circuit is None:
                    continue
                n = circuit.num_qubits
                q_vol, b_vol, capacity = run_point(circuit, model)
                speedup = b_vol / q_vol
                print(f"  n={n} capacity={capacity} qldpc_vol={q_vol:.1f} baseline_vol={b_vol:.1f} speedup={speedup:.3f}")
                writer.writerow([family, n, capacity, q_vol, b_vol, speedup])
                f.flush()

    print(f"\nwrote results to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
