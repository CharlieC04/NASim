"""
Generates LARGER versions of the real Clifford+T circuits in this
directory (qrom, mcx, hb_sel), aiming to cross >=100 qubits after
decomposition -- for the circuit-size scaling experiment
(examples/qldpc_scaling_experiment.py).

MUST be run in an isolated environment with cirq-ft installed, NOT the
main repo venv: cirq-ft requires numpy<2.0, which directly conflicts
with qiskit's numpy>=2.0 requirement used throughout the rest of this
repo. See tests/fixtures/cirq_ft_tgate/README.md for why, and the
bottom of this file / the task instructions for exact setup commands.

The parameter choices below are REASONED ESTIMATES extrapolated from the
smaller circuits already committed in this directory (qrom: 32 entries,
5-bit -> 14 decomposed qubits; mcx: 6 controls -> 12 decomposed qubits;
hb_sel: 2x2 lattice -> 33 decomposed qubits), not confirmed at this
scale -- this script prints the actual decomposed qubit count for each
size point as it runs, so check the printed counts and adjust the
sweep lists below (grep for "SWEEP") if a particular point lands short
of or far past 100.
"""

import numpy as np
import cirq
import cirq_ft

_TARGETS_1Q = {
    "H": cirq.unitary(cirq.H),
    "S": cirq.unitary(cirq.S),
    "Sdg": cirq.unitary(cirq.S**-1),
    "X": cirq.unitary(cirq.X),
    "Y": cirq.unitary(cirq.Y),
    "Z": cirq.unitary(cirq.Z),
    "T": cirq.unitary(cirq.T),
    "Tdg": cirq.unitary(cirq.T**-1),
    "SqrtX": cirq.unitary(cirq.X**0.5),
    "SqrtXdg": cirq.unitary(cirq.X**-0.5),
    "SqrtY": cirq.unitary(cirq.Y**0.5),
    "SqrtYdg": cirq.unitary(cirq.Y**-0.5),
}
_TARGETS_2Q = {
    "CNOT": cirq.unitary(cirq.CNOT),
    "CZ": cirq.unitary(cirq.CZ),
}
_CANONICAL = {
    "H": cirq.H, "S": cirq.S, "Sdg": cirq.S**-1, "X": cirq.X, "Y": cirq.Y, "Z": cirq.Z,
    "T": cirq.T, "Tdg": cirq.T**-1, "SqrtX": cirq.X**0.5, "SqrtXdg": cirq.X**-0.5,
    "SqrtY": cirq.Y**0.5, "SqrtYdg": cirq.Y**-0.5, "CNOT": cirq.CNOT, "CZ": cirq.CZ,
}


def _is_dropped(gate) -> bool:
    return isinstance(gate, (cirq.MeasurementGate, cirq.ResetChannel, cirq.GlobalPhaseGate))


def classify_unitary(op, tol: float = 1e-6):
    if len(op.qubits) == 1:
        targets = _TARGETS_1Q
    elif len(op.qubits) == 2:
        targets = _TARGETS_2Q
    else:
        return None
    try:
        u = cirq.unitary(op)
    except (TypeError, ValueError):
        return None
    for name, target in targets.items():
        if cirq.allclose_up_to_global_phase(u, target, atol=tol):
            return name
    return None


def _keep(op):
    if _is_dropped(op.gate):
        return True
    if isinstance(op, cirq.ClassicallyControlledOperation):
        return True
    return classify_unitary(op) is not None


def decompose_to_clifford_t(circuit: cirq.Circuit) -> cirq.Circuit:
    return cirq.Circuit(cirq.decompose(circuit, keep=_keep))


def gate_counts(circuit: cirq.Circuit) -> dict:
    counts: dict = {}
    dropped = 0
    for moment in circuit:
        for op in moment.operations:
            g = op
            if isinstance(op, cirq.ClassicallyControlledOperation):
                g = op.without_classical_controls()
            if _is_dropped(g.gate):
                dropped += 1
                continue
            key = classify_unitary(g)
            counts[key] = counts.get(key, 0) + 1
    counts["_dropped_measure_reset"] = dropped
    return counts


def export_qasm(circuit: cirq.Circuit, out_path: str) -> None:
    clean_ops = []
    for moment in circuit:
        for op in moment.operations:
            g = op
            if isinstance(op, cirq.ClassicallyControlledOperation):
                g = op.without_classical_controls()
            if _is_dropped(g.gate):
                continue
            name = classify_unitary(g)
            clean_ops.append(_CANONICAL[name].on(*g.qubits))
    qasm_text = cirq.qasm(cirq.Circuit(clean_ops))
    with open(out_path, "w") as f:
        f.write(qasm_text)


def build_qrom(n_entries: int, n_target_arrays: int, bits_per_array: int):
    # Multiple parallel data arrays (QROM.build(*data) accepts several)
    # -> multiple target registers, each bits_per_array wide, pushing
    # qubit count up independent of n_entries. Values are arbitrary
    # (just need to need bits_per_array bits), not meaningful data.
    rng = np.random.default_rng(0)
    data_arrays = [
        rng.integers(0, 2**bits_per_array, size=n_entries, dtype=np.int64)
        for _ in range(n_target_arrays)
    ]
    qrom = cirq_ft.QROM.build(*data_arrays)
    qubits = cirq.LineQubit.range(cirq.num_qubits(qrom))
    return cirq.Circuit(qrom.on(*qubits))


def build_mcx(num_controls: int):
    gate = cirq_ft.MultiControlPauli(cvs=[1] * num_controls, target_gate=cirq.X)
    qubits = cirq.LineQubit.range(cirq.num_qubits(gate))
    return cirq.Circuit(gate.on(*qubits))


def build_select_hubbard(x_dim: int, y_dim: int):
    sel = cirq_ft.SelectHubbard(x_dim=x_dim, y_dim=y_dim)
    qubits = cirq.LineQubit.range(cirq.num_qubits(sel))
    return cirq.Circuit(sel.on(*qubits))


if __name__ == "__main__":
    # SWEEP: each tuple is (label, builder()). Adjust parameters here if
    # the printed decomposed-qubit count for a size point isn't close to
    # what you need -- see this file's docstring.
    sweeps = {
        "qrom": [
            ("qrom_s1", lambda: build_qrom(n_entries=32, n_target_arrays=2, bits_per_array=20)),
            ("qrom_s2", lambda: build_qrom(n_entries=64, n_target_arrays=2, bits_per_array=40)),
            ("qrom_s3", lambda: build_qrom(n_entries=64, n_target_arrays=3, bits_per_array=40)),
        ],
        "mcx": [
            ("mcx_s1", lambda: build_mcx(num_controls=20)),
            ("mcx_s2", lambda: build_mcx(num_controls=50)),
            ("mcx_s3", lambda: build_mcx(num_controls=90)),
        ],
        "hb_sel": [
            ("hb_sel_s1", lambda: build_select_hubbard(x_dim=3, y_dim=3)),
            ("hb_sel_s2", lambda: build_select_hubbard(x_dim=4, y_dim=4)),
            ("hb_sel_s3", lambda: build_select_hubbard(x_dim=5, y_dim=5)),
        ],
    }

    out_dir = "tests/fixtures/cirq_ft_tgate"
    for family, points in sweeps.items():
        for label, builder in points:
            circuit = builder()
            print(f"{label}: raw qubits = {len(circuit.all_qubits())}")
            decomposed = decompose_to_clifford_t(circuit)
            counts = gate_counts(decomposed)
            n_qubits = len(decomposed.all_qubits())
            t_count = counts.get("T", 0) + counts.get("Tdg", 0)
            print(f"{label}: decomposed qubits = {n_qubits}, T-count = {t_count}, counts = {counts}")
            out_path = f"{out_dir}/{label}.qasm"
            export_qasm(decomposed, out_path)
            print(f"{label}: wrote {out_path}\n")
