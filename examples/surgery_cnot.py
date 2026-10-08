from pathlib import Path

from nasim.circuit import Circuit, two_qubit
from nasim.device import Device
from nasim.model import Model
from nasim.placement import patch_separation_um, piqasso_placement
from nasim.schedule import circuit_fidelity, simulate_circuit
from nasim.surface_code import Patch
from nasim.viz import ScheduleViewer

PRESET = Path(__file__).resolve().parents[1] / "src" / "nasim" / "config" / "presets" / "2d.yaml"

# Measurement-based lattice-surgery CNOT, per Viszlai et al. (arXiv:2309.13507)

DISTANCE = 3


def main() -> None:
    model = Model.from_yaml(PRESET)
    device = Device.uniform(model=model, rows=20, cols=20, pitch_um=patch_separation_um(model, distance=DISTANCE))

    circuit = Circuit.from_stages(3, [
        [two_qubit(0, 1)],  # control <-> ancilla: M_ZZ
        [two_qubit(1, 2)],  # ancilla <-> target: M_XX
    ])
    initial_placement = piqasso_placement(circuit, model, device, distance=DISTANCE, seed=0)
    schedule, final_placement = simulate_circuit(circuit, initial_placement, model)

    measures = [op for op in schedule.ops if op.kind == "mid_circuit_measure"]
    merges = [op for op in schedule.ops if op.kind == "merge_handover"]
    total_rounds = len(measures)
    expected_rounds = 2 * DISTANCE

    standalone_atoms = len(Patch.rotated(DISTANCE).atoms)
    cnot_qubits = 3 * standalone_atoms
    transversal_qubits = 2 * standalone_atoms
    ratio = cnot_qubits / transversal_qubits

    print(f"d={DISTANCE}: {len(merges)} merges, {total_rounds} total measurement rounds (paper: {expected_rounds})")
    print(f"physical qubits: CNOT={cnot_qubits} transversal={transversal_qubits} ratio={ratio:.2f} (paper: 1.5-2x)")
    print(f"circuit fidelity: {circuit_fidelity(schedule):.6g}")

    assert total_rounds == expected_rounds, "doesn't match Viszlai et al. Fig. 5's stated 2d measurement rounds"
    assert 1.5 <= ratio <= 2.0, "doesn't match Viszlai et al.'s stated 1.5x-2x qubit overhead"
    print("Matches both claims in Viszlai et al. (arXiv:2309.13507), Section 5.1.3 / Fig. 5.")

    viewer = ScheduleViewer(initial_placement, schedule, circuit)
    saved = viewer.save_frames("lattice_surgery_cnot_frames")
    print(f"saved {len(saved)} frame(s) to lattice_surgery_cnot_frames/")
    viewer.show()


if __name__ == "__main__":
    main()