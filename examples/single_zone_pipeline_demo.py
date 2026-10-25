import dataclasses
import random
import time
from pathlib import Path

from nasim.circuit import Circuit, one_qubit, two_qubit, magic_consume
from nasim.device import Device
from nasim.model import Model
from nasim.placement import zap_placement
from nasim.resource import Factory
from nasim.schedule import circuit_fidelity, simulate_circuit

# Exercises the real pipeline (zap_placement -> simulate_circuit -> Factory
# delivery -> circuit_fidelity) on the single-zone device at FT-Weave's own
# d=7 scale: Ncnot=64, n_1q=2624, Nt=2528 (output/evaluation/fidelity/
# t_cultivation_fidelity_results.csv) - not just the small sanity-check
# circuit from before, and not just the fidelity formula fed synthetic
# counts (test_ftweave_circuit_fidelity_validation.py checks that).
#
# This isn't FT-Weave's literal TFIM circuit (that needs Rz-rotation
# synthesis we don't model) - it's a circuit with matching AGGREGATE gate
# counts, to stress-test the scheduler/placement/magic-delivery pipeline
# at real scale, and to roughly match their fidelity_total using their own
# LOGICAL_SE_INTERVAL idle-syndrome-extraction cadence (src/t_cultivation/
# config.py), not a literal reproduction of their compiled schedule.

PRESET = Path(__file__).resolve().parents[1] / "src" / "nasim" / "config" / "presets" / "2d_nozone.yaml"
DISTANCE = 7
NUM_QUBITS = 16
NUM_FACTORIES = 16

TARGET_NCNOT = 64
TARGET_N1Q = 2624
TARGET_NT = 2528

ROUNDS = TARGET_N1Q // NUM_QUBITS          # 164
T_ROUNDS = TARGET_NT // NUM_QUBITS         # 158
CNOT_ROUNDS = (TARGET_NCNOT // (NUM_QUBITS // 2))  # 8, each a disjoint-pair layer


def build_circuit() -> Circuit:
    stages = []
    for r in range(ROUNDS):
        stages.append([one_qubit(q) for q in range(NUM_QUBITS)])
        if r < T_ROUNDS:
            stages.append([magic_consume(q) for q in range(NUM_QUBITS)])
        if r < CNOT_ROUNDS:
            stages.append([two_qubit(i, i + 1) for i in range(0, NUM_QUBITS, 2)])
    return Circuit.from_stages(NUM_QUBITS, stages)


def main() -> None:
    # FT-Weave's own LOGICAL_SE_INTERVAL (src/t_cultivation/config.py): idle
    # logical qubits are syndrome-extracted every 10 cycles, not every cycle.
    model = dataclasses.replace(Model.from_yaml(PRESET), logical_se_interval=10)
    device = Device.from_yaml(PRESET, model=model)
    circuit = build_circuit()

    t0 = time.monotonic()
    placement = zap_placement(circuit, model, device, distance=DISTANCE)
    factories = [
        Factory(resource_id=-(i + 1), model=model, rng=random.Random(i), distance=DISTANCE, num_subfactories=3)
        for i in range(NUM_FACTORIES)
    ]

    schedule, final = simulate_circuit(circuit, placement, model, device=device, factories=factories)
    f = circuit_fidelity(schedule, d=DISTANCE)
    elapsed = time.monotonic() - t0

    print(f"wall-clock compile+simulate time: {elapsed:.2f}s")
    print(f"Ncnot={schedule.Ncnot} (target {TARGET_NCNOT})")
    print(f"Nt={schedule.Nt} (target {TARGET_NT})")
    n_1q = sum(len(op.detail["qubits"]) for op in schedule.ops if op.kind == "1q_stage")
    print(f"n_1q={n_1q} (target {TARGET_N1Q})")
    print(f"Nh={schedule.Nh}  clock_us={schedule.clock_us:.1f}")
    print(f"circuit_fidelity={f:.6e}")

    assert schedule.Ncnot == TARGET_NCNOT
    assert schedule.Nt == TARGET_NT
    assert n_1q == TARGET_N1Q
    assert 0.0 <= f <= 1.0

    ent_zone = device.zones[0]
    resident = sum(1 for p in final.patches if ent_zone.contains(p.anchor))
    print(f"{resident}/{len(final.patches)} patches resident in the single zone at the end")

    print("OK: real pipeline holds up at FT-Weave's own d=7 gate-count scale")


if __name__ == "__main__":
    main()
