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

# A single physical Factory runs both the check and escape stages itself,
# modeled on FT-Weave's actual reference implementation (src/ds/
# device_state_t.py's TFactory, not just the paper): a fixed-duration
# check attempt with num_subfactories parallel Bernoulli lines, then one
# escape attempt. Delivery to the requesting qubit is a routing-aware
# bipartite match, followed by a coin flip for the standard gate-
# teleportation measurement byproduct (cultivation_teleportation_success_
# prob) - on the "wrong branch" a corrective single-qubit gate is
# scheduled. See nasim.schedule.magic._resolve_magic_requests for the
# full lifecycle, including check-stage redistribution across factories.

DISTANCE = 3


def main() -> None:
    model = Model.from_yaml(PRESET)
    pitch = patch_separation_um(model, distance=DISTANCE)

    # gap_um must exceed pitch, not just equal it - see
    # test_device.py's test_zone_of_returns_none_in_the_gap_between_zones.
    device = Device.zoned(
        model=model,
        storage_rows=6, storage_cols=6, storage_pitch_um=pitch,
        entanglement_rows=3, entanglement_cols=5, entanglement_pitch=pitch,
        gap_um=2 * pitch,
    )

    circuit = Circuit.from_stages(3, [
        [one_qubit(0), one_qubit(1), one_qubit(2)],
        [magic_consume(0)],       # q0 gets a T gate
        [two_qubit(1, 2)],        # an ordinary 2Q gate elsewhere
        [magic_consume(1)],       # q1 gets a T gate - fresh check+escape cycle
    ])

    placement = zap_placement(circuit, model, device, distance=DISTANCE)
    factories = [Factory(resource_id=-1, model=model, rng=random.Random(1), distance=DISTANCE)]

    schedule, final_placement = simulate_circuit(circuit, placement, model, device=device, factories=factories)

    waits = [op for op in schedule.ops if op.kind == "await_resource"]
    merges = [op for op in schedule.ops if op.kind == "merge_handover"]
    corrections = [op for op in schedule.ops if op.kind == "correction_gate"]

    print(f"clock={schedule.clock_us:.1f}us  Nh={schedule.Nh}  Nxtalk={schedule.Nxtalk}")
    print(f"{len(waits)} wait(s) for check/escape, {len(merges)} merge(s), "
          f"{len(corrections)} RUS correction gate(s) recorded")
    for op in waits:
        print(f"  waited {op.duration_us:.1f}us for qubits {op.detail['qubits']}")
    print(f"circuit fidelity: {circuit_fidelity(schedule, d=DISTANCE):.6g}")

    ent_zone = next(z for z in device.zones if z.kind == "entanglement")
    storage_zone = next(z for z in device.zones if z.kind == "storage")
    for p in final_placement.patches:
        kind = "factory" if p.qubit < 0 else "qubit"
        loc = "entanglement" if ent_zone.contains(p.anchor) else "storage" if storage_zone.contains(p.anchor) else "?"
        label = f"factory F{-p.qubit}" if p.qubit < 0 else f"q{p.qubit}"
        print(f"  {label} ({kind}): {loc}")

    viewer = ScheduleViewer(placement, schedule, circuit, device=device, final=final_placement)
    saved = viewer.save_frames("magic_consume_demo_frames")
    print(f"saved {len(saved)} frame(s) to magic_consume_demo_frames/")
    viewer.save_gif("magic_consume_demo.gif", fps=1.0)
    viewer.show()


if __name__ == "__main__":
    main()
