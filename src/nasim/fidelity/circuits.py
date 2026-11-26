from __future__ import annotations

import numpy as np
import stim

from nasim.qldpc import BicycleTerm, GenBicycleCode

def gen_bicycle_memory_circuit(code: GenBicycleCode, *, rounds: int, p: float, basis: str = "Z") -> stim.Circuit:

    """
    Noise model follows qSieve. Data qubits start in |0>^N for Z basis, |+>^N for X
    """

    lm = code.lm
    n = code.n
    DATA = list(range(n))
    XANC = list(range(n, n+lm))
    ZANC = list(range(n+lm, n+2*lm))

    def term_targets(term: BicycleTerm, block_ofs: int) -> list[int]:
        mat = code._term_matrix(term)
        return [int(np.nonzero(mat[r])[0][0]) + block_ofs for r in range(lm)]

    def neg(term: BicycleTerm) -> BicycleTerm:
        return BicycleTerm(p=(-term.p) % code.l, q=(-term.q) % code.m)

    x_layers = [term_targets(t, 0) for t in code.a_terms] + [term_targets(t, lm) for t in code.b_terms]
    z_layers = [term_targets(neg(t), 0) for t in code.b_terms] + [term_targets(neg(t), lm) for t in code.a_terms]

    circuit = stim.Circuit()
    circuit.append("R", DATA)
    if basis == "X":
        circuit.append("H", DATA)
    circuit.append("X_ERROR" if basis == "Z" else "Z_ERROR", DATA, p)

    determ_at_r0 = "Z" if basis == "Z" else "X"

    for r in range(rounds):
        circuit.append("R", XANC + ZANC)
        circuit.append("X_ERROR", XANC + ZANC, p)
        circuit.append("H", XANC)

        for layer in x_layers:
            pairs = [q for anc, dq in zip(XANC, layer) for q in (anc, dq)]
            circuit.append("CNOT", pairs)
            circuit.append("DEPOLARIZE2", pairs, p)
        for layer in z_layers:
            pairs = [q for anc, dq in zip(ZANC, layer) for q in (dq, anc)]
            circuit.append("CNOT", pairs)
            circuit.append("DEPOLARIZE2", pairs, p)

        circuit.append("H", XANC)
        circuit.append("X_ERROR", XANC + ZANC, p)
        circuit.append("MR", XANC + ZANC)

        n_x, n_z = len(XANC), len(ZANC)
        n_anc = n_x + n_z
        for i in range(n_x):
            if r == 0 and determ_at_r0 != "X": continue

            rec_now = -n_anc + i
            targets = [stim.target_rec(rec_now)]
            if r > 0: targets.append(stim.target_rec(rec_now - n_anc))
            circuit.append("DETECTOR", targets, [0])
        for j in range(n_z):
            if r == 0 and determ_at_r0 != "Z": continue

            rec_now = -n_anc + n_x + j
            targets = [stim.target_rec(rec_now)]
            if r > 0: targets.append(stim.target_rec(rec_now - n_anc))
            circuit.append("DETECTOR", targets, [1])

    circuit.append("X_ERROR" if basis == "Z" else "Z_ERROR", DATA, p)
    if basis == "X":
        circuit.append("H", DATA)
    circuit.append("M", DATA)

    n_x, n_z = len(XANC), len(ZANC)
    n_anc = n_x + n_z
    if determ_at_r0 == "Z":
        for r in range(n_z):
            support = {z_layers[k][r] for k in range(6)}
            rec_last = (-n_anc + n_x + r) - n
            targets = [stim.target_rec(-n + q) for q in sorted(support)]
            targets.append(stim.target_rec(rec_last))
            circuit.append("DETECTOR", targets, [1])
    else:
        for r in range(n_x):
            support = {x_layers[k][r] for k in range(6)}
            rec_last = (-n_anc + r) - n
            targets = [stim.target_rec(-n + q) for q in sorted(support)]
            targets.append(stim.target_rec(rec_last))
            circuit.append("DETECTOR", targets, [0])

    for i, (x_supp, z_supp) in enumerate(code.logical_operators()):
        support = z_supp if basis == "Z" else x_supp
        targets = [stim.target_rec(-n + q) for q in sorted(support)]
        circuit.append("OBSERVABLE_INCLUDE", targets, i)

    return circuit

def surface_code_memory_circuit(*, distance: int, rounds: int, p: float, basis: str = "Z") -> stim.Circuit:

    task = "surface_code:rotated_memory_z" if basis == "Z" else "surface_code:rotated_memory_x"
    return stim.Circuit.generated(
        task,
        rounds=rounds,
        distance=distance,
        after_clifford_depolarization=p,
        before_round_data_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p
    )