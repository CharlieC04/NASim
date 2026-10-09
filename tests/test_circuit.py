from nasim.circuit import Circuit, GateType, one_qubit, two_qubit


def _types(stage):
    return {g.type for g in stage}


def test_stages_are_always_pure():
    circuit = Circuit.from_gates(4, [
        one_qubit(0), one_qubit(1),
        two_qubit(0, 1),
        one_qubit(2), two_qubit(1, 2),
        one_qubit(3),
    ])
    for stage in circuit.stages:
        assert len(_types(stage)) == 1


def test_disjoint_two_qubit_gates_pack_into_the_same_checkpoint():
    g01 = two_qubit(0, 1)
    g23 = two_qubit(2, 3)
    circuit = Circuit.from_gates(4, [g01, g23])
    assert len(circuit.stages) == 1
    assert set(circuit.stages[0]) == {g01, g23}


def test_unrelated_one_qubit_gate_runs_asap_not_in_program_order():
    # qubit 3 has no dependency on the qubit-0/1/2 chain at all, so its 1Q
    # gate should land in the very first available stage even though it's
    # listed last.
    g01 = two_qubit(0, 1)
    g12 = two_qubit(1, 2)
    g3 = one_qubit(3)
    circuit = Circuit.from_gates(4, [g01, g12, g3])

    assert circuit.stages[0] == (g3,)
    assert circuit.stages[1] == (g01,)
    assert circuit.stages[2] == (g12,)


def test_unrelated_gap_fill_does_not_separate_adjacent_two_qubit_checkpoints():
    # one_qubit(2) is interleaved in program order between qubit 1's two
    # 2Q gates, but it has no dependency forcing it to sit between them
    # (it's the first touch of qubit 2) - ASAP-separate should keep
    # two_qubit(0,1) and two_qubit(1,2) in adjacent stages regardless.
    g01 = two_qubit(0, 1)
    g2 = one_qubit(2)
    g12 = two_qubit(1, 2)
    circuit = Circuit.from_gates(3, [g01, g2, g12])

    idx01 = next(i for i, s in enumerate(circuit.stages) if g01 in s)
    idx12 = next(i for i, s in enumerate(circuit.stages) if g12 in s)
    assert idx12 - idx01 == 1


def test_sequential_one_qubit_gates_on_the_same_qubit_serialize():
    gates = [one_qubit(0), one_qubit(0), one_qubit(0)]
    circuit = Circuit.from_gates(1, gates)
    assert len(circuit.stages) == 3
    assert [s[0] for s in circuit.stages] == gates


def test_from_gates_returns_a_circuit_with_tuple_of_tuple_stages():
    circuit = Circuit.from_gates(2, [one_qubit(0), two_qubit(0, 1)])
    assert isinstance(circuit, Circuit)
    assert circuit.num_qubits == 2
    assert isinstance(circuit.stages, tuple)
    assert all(isinstance(s, tuple) for s in circuit.stages)
