from __future__ import annotations

from pathlib import Path
from qiskit import QuantumCircuit, transpile
from qiskit_qasm3_import import parse

from nasim.circuit import Gate, one_qubit, two_qubit

def load_transpiled_circuit(path: str | Path) -> tuple[int, list[Gate]]:

    """
    Use ZAP's transpilation method
    """

    path = Path(path)
    qasm_str = path.read_text()

    if "OPENQASM 2" in qasm_str:
        circuit = QuantumCircuit.from_qasm_str(qasm_str)
    elif "OPENQASM 3" in qasm_str:
        circuit = parse(qasm_str)
    else:
        raise ValueError("unsupported code")

    while circuit.data and circuit.data[-1].operation.name == "swap": circuit.data.pop()

    n_pre = circuit.num_qubits
    if n_pre <= 24: opt_level = 3
    elif n_pre <= 64: opt_level = 2
    elif n_pre <= 128: opt_level = 1
    else: opt_level = 0

    cz_cirq = transpile(
        circuit,
        basis_gates=["cz", "id", "u2", "u1", "u3"],
        optimization_level=opt_level,
        seed_transpiler=0
    )

    gates: list[Gate] = []
    for instr in cz_cirq.data:
        if instr.operation.num_qubits == 2:
            qa = cz_cirq.find_bit(instr.qubits[0]).index
            qb = cz_cirq.find_bit(instr.qubits[1]).index
            gates.append(two_qubit(qa, qb))
        elif instr.operation.name not in ("measure", "barrier"):
            q = cz_cirq.find_bit(instr.qubits[0]).index
            gates.append(one_qubit(q))

    return cz_cirq.num_qubits, gates