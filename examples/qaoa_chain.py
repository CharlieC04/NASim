from pathlib import Path

from nasim.circuit import Circuit, one_qubit, two_qubit
from nasim.device import Device
from nasim.model import Model
from nasim.placement import patch_separation_um, piqasso_placement
from nasim.schedule import circuit_fidelity, simulate_circuit
from nasim.viz import ScheduleViewer

PRESET = Path(__file__).resolve().parents[1] / "src" / "nasim" / "config" / "presets" / "2d.yaml"

N_QUBITS = 6
P_ROUNDS = 2


def qaoa_chain_circuit(n_qubits: int, p_rounds: int) -> Circuit:

    """
    QAOA/Ising brick-wall ansatz on a path graph: each round is a mixer
    layer (1Q gate per qubit) followed by the even-edge ZZ layer then the
    odd-edge ZZ layer, so every 2Q stage is a set of disjoint, independently
    routable pairs - exactly the case the concurrent-batching, parking and
    routing-aware placement work in this session targets.
    """

    stages = []
    for _ in range(p_rounds):
        stages.append([one_qubit(q) for q in range(n_qubits)])
        stages.append([two_qubit(i, i + 1) for i in range(0, n_qubits - 1, 2)])
        stages.append([two_qubit(i, i + 1) for i in range(1, n_qubits - 1, 2)])
    return Circuit.from_stages(n_qubits, stages)


def main() -> None:
    model = Model.from_yaml(PRESET)
    device = Device.uniform(model=model, rows=20, cols=20, pitch_um=patch_separation_um(model, distance=3))

    circuit = qaoa_chain_circuit(N_QUBITS, P_ROUNDS)
    initial_placement = piqasso_placement(circuit, model, device, distance=3, seed=0)

    schedule, final_placement = simulate_circuit(circuit, initial_placement, model)

    move_frames = [op for op in schedule.ops if op.kind == "move_frame"]
    merges = [op for op in schedule.ops if op.kind == "merge_handover"]
    print(f"{len(schedule.ops)} raw ops, {len(move_frames)} move frames, {len(merges)} merges")
    print(
        f"Ng1={schedule.Ng1} Ng2={schedule.Ng2} Nh={schedule.Nh} Nmeas={schedule.Nmeas} "
        f"clock={schedule.clock_us:.2f}us"
    )
    print(f"circuit fidelity: {circuit_fidelity(schedule):.6g}")

    viewer = ScheduleViewer(initial_placement, schedule, circuit)
    print(f"{len(viewer.groups)} display frames (collapsed from {len(schedule.ops)} raw ops)")
    saved = viewer.save_frames("qaoa_chain_frames")
    print(f"saved {len(saved)} frame(s) to qaoa_chain_frames/")
    viewer.save_gif("qaoa_chain.gif")
    viewer.show()


if __name__ == "__main__":
    main()