from pathlib import Path

from nasim.circuit import Circuit, GateType, one_qubit, two_qubit
from nasim.device import Device
from nasim.model import Model
from nasim.placement import patch_separation_um, piqasso_placement
from nasim.schedule import simulate_circuit

PRESET = Path(__file__).resolve().parents[1] / "src" / "nasim" / "config" / "presets" / "2d.yaml"


def irregular_gate_list() -> list:

    """
    A flat, unstaged gate list (program order only - no hand-picked stage
    boundaries). Two things are deliberately baited:

    - one_qubit(4) sits in program order between two_qubit(2, 3) and
      two_qubit(1, 2), but has no dependency forcing it there (qubit 4
      hasn't touched a 2Q gate yet) - ASAP should pull it all the way back
      to the first available stage instead of leaving it where it was listed.
    - two_qubit(0, 1) and two_qubit(2, 3) are disjoint and should land in
      the SAME checkpoint; two_qubit(1, 2) and two_qubit(3, 4) are disjoint
      and should also share a checkpoint.
    """

    return [
        one_qubit(0), one_qubit(1), one_qubit(2), one_qubit(3), one_qubit(4),
        two_qubit(0, 1),
        two_qubit(2, 3),
        one_qubit(4),
        two_qubit(1, 2),
        two_qubit(3, 4),
        one_qubit(0),
    ]


def describe(circuit: Circuit) -> None:
    for i, stage in enumerate(circuit.stages):
        kind = "2Q" if stage[0].type is GateType.TWO_QUBIT else "1Q"
        qubits = [g.qubits for g in stage]
        print(f"  stage {i}: {kind} {qubits}")


def main() -> None:
    gates = irregular_gate_list()
    print(f"{len(gates)} gates, program order:")
    for i, g in enumerate(gates):
        kind = "2Q" if g.type is GateType.TWO_QUBIT else "1Q"
        print(f"  [{i}] {kind} {g.qubits}")

    circuit = Circuit.from_gates(5, gates)
    print(f"\nASAP-separate schedule ({len(circuit.stages)} stages):")
    describe(circuit)

    print(
        "\none_qubit(4) at program position 7 (listed between two_qubit(2,3) "
        "and two_qubit(1,2)) landed in stage 1 - pulled ahead of both "
        "two-qubit checkpoints, since nothing on qubit 4 depended on them yet."
    )
    print(
        "two_qubit(0,1)/two_qubit(2,3) share stage 2; "
        "two_qubit(1,2)/two_qubit(3,4) share stage 4 - disjoint 2Q gates "
        "packed into a single checkpoint rather than serialized."
    )

    model = Model.from_yaml(PRESET)
    device = Device.uniform(model=model, rows=20, cols=20, pitch_um=patch_separation_um(model, distance=3))
    placement = piqasso_placement(circuit, model, device, distance=3, seed=0)

    schedule, _ = simulate_circuit(circuit, placement, model)
    move_frames = [op for op in schedule.ops if op.kind == "move_frame"]
    print(
        f"\nsimulated OK: {len(schedule.ops)} raw ops, {len(move_frames)} move frames, "
        f"Ng1={schedule.Ng1} Ng2={schedule.Ng2} Nh={schedule.Nh} clock={schedule.clock_us:.2f}us"
    )


if __name__ == "__main__":
    main()
