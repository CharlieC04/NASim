import random
from pathlib import Path

from nasim.circuit import Circuit, one_qubit, two_qubit, magic_consume
from nasim.device import Device
from nasim.model import Model
from nasim.placement import patch_separation_um, zap_placement
from nasim.resource import Factory
from nasim.schedule import circuit_fidelity, simulate_circuit
from nasim.viz import ScheduleViewer

PRESET = Path(__file__).resolve().parents[1] / "src" / "nasim" / "config" / "presets" / "2d.yaml"

# Multiple factories + concurrent magic_consume requests exercises
# nasim.schedule.magic's event-driven resolver, modeled on FT-Weave's
# actual reference implementation: each Factory runs its own check stage
# (num_subfactories parallel Bernoulli lines, fixed duration) then its own
# escape stage. Factories whose check attempts complete at the same
# instant are batched and redistributed - a routing-aware bipartite match
# moves spare successes from factories with >=2 to factories with 0,
# charging only a time delay (no merge/split - see
# nasim.resource.Factory.num_spare_successes). Delivery to a requesting
# qubit is the same routing-aware match, not first-come-first-served,
# followed by a coin flip for the gate-teleportation measurement
# byproduct that sometimes needs a corrective single-qubit gate.

DISTANCE = 3
NUM_FACTORIES = 3
NUM_QUBITS = 5


def main() -> None:
    model = Model.from_yaml(PRESET)
    pitch = patch_separation_um(model, distance=DISTANCE)

    # gap_um must exceed pitch, not just equal it - see
    # test_device.py's test_zone_of_returns_none_in_the_gap_between_zones.
    # Generously sized Ez so several qubits and several factories can be
    # concurrently resident.
    device = Device.zoned(
        model=model,
        storage_rows=8, storage_cols=8, storage_pitch_um=pitch,
        entanglement_rows=6, entanglement_cols=8, entanglement_pitch=pitch,
        gap_um=2 * pitch,
    )

    circuit = Circuit.from_stages(NUM_QUBITS, [
        [one_qubit(q) for q in range(NUM_QUBITS)],
        # 5 concurrent requests, 3 factories: later ones wait for an
        # earlier delivery's factory to free up and restart.
        [magic_consume(q) for q in range(NUM_QUBITS)],
        [two_qubit(0, 1), two_qubit(2, 3)],
        # A second round of requests re-uses the now-recultivating
        # factories; fewer requests than factories this time.
        [magic_consume(0), magic_consume(2), magic_consume(4)],
    ])

    placement = zap_placement(circuit, model, device, distance=DISTANCE)
    # num_subfactories > 1 so a factory can occasionally have a spare
    # success to redistribute - with the yaml default (1), no factory can
    # ever have more than 1 success, so redistribution is structurally
    # impossible and this demo would never show it.
    factories = [
        Factory(resource_id=-(i + 1), model=model, rng=random.Random(i), distance=DISTANCE, num_subfactories=3)
        for i in range(NUM_FACTORIES)
    ]

    schedule, final_placement = simulate_circuit(circuit, placement, model, device=device, factories=factories)

    waits = [op for op in schedule.ops if op.kind == "await_resource"]
    deliveries = [op for op in schedule.ops if op.kind == "merge_handover" and sum(q < 0 for q in op.detail["qubits"]) == 1]
    redistributions = [op for op in schedule.ops if op.kind == "redistribution_delay"]
    corrections = [op for op in schedule.ops if op.kind == "correction_gate"]

    print(f"clock={schedule.clock_us:.1f}us  Nh={schedule.Nh}  Nxtalk={schedule.Nxtalk}")
    print(f"{NUM_FACTORIES} factories, {NUM_QUBITS + 3} magic_consume requests across 2 stages")
    print(f"{len(waits)} wait(s), {len(redistributions)} check-stage redistribution(s), "
          f"{len(deliveries)} escape->qubit delivery(ies), {len(corrections)} RUS correction gate(s)\n")

    print("deliveries, in schedule order (factory -> qubit, routing-aware assignment):")
    for op in deliveries:
        data_qubit = next(q for q in op.detail["qubits"] if q >= 0)
        factory_qubit = next(q for q in op.detail["qubits"] if q < 0)
        print(f"  t={op.start_us:7.1f}us  F{-factory_qubit} -> q{data_qubit}")

    print("\ncheck-stage redistributions (donor -> receiver, time cost only):")
    for op in redistributions:
        print(f"  t={op.start_us:7.1f}us  F{-op.detail['donor']} -> F{-op.detail['receiver']}  ({op.duration_us:.1f}us)")

    print(f"\ncircuit fidelity: {circuit_fidelity(schedule, d=DISTANCE):.6g}")

    ent_zone = next(z for z in device.zones if z.kind == "entanglement")
    storage_zone = next(z for z in device.zones if z.kind == "storage")
    print("\nfinal placement:")
    for p in sorted(final_placement.patches, key=lambda p: p.qubit):
        kind = "factory" if p.qubit < 0 else "qubit"
        loc = "entanglement" if ent_zone.contains(p.anchor) else "storage" if storage_zone.contains(p.anchor) else "?"
        label = f"factory F{-p.qubit}" if p.qubit < 0 else f"q{p.qubit}"
        print(f"  {label} ({kind}): {loc}")

    viewer = ScheduleViewer(placement, schedule, circuit, device=device, final=final_placement)
    saved = viewer.save_frames("magic_multi_factory_demo_frames")
    print(f"\nsaved {len(saved)} frame(s) to magic_multi_factory_demo_frames/")
    viewer.save_gif("magic_multi_factory_demo.gif", fps=1.0)
    viewer.show()


if __name__ == "__main__":
    main()
