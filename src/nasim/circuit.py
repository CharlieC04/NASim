from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import math

class GateType(Enum):
    ONE_QUBIT = "1q"
    TWO_QUBIT = "2q"

@dataclass(frozen=True)
class Gate:

    type: GateType
    qubits: tuple[int, ...]

def one_qubit(qubit: int) -> Gate:
    return Gate(GateType.ONE_QUBIT, (qubit,))

def two_qubit(a: int, b: int) -> Gate:
    return Gate(GateType.TWO_QUBIT, (a,b))

@dataclass(frozen=True)
class Circuit:

    """
    A logical circuit in stages, where gates in a stage can be executed concurrently
    """

    num_qubits: int
    stages: tuple[tuple[Gate, ...], ...]

    @classmethod
    def from_stages(cls, num_qubits: int, stages: list[list[Gate]]) -> Circuit:
        return cls(num_qubits, tuple(tuple(stage) for stage in stages))

    @classmethod
    def from_gates(cls, num_qubits: int, gates: list[Gate]) -> Circuit:

        """
        ASAP-separate scheduling (ZAC)
        """

        two_qubit_gates = [g for g in gates if g.type is GateType.TWO_QUBIT]

        cursor_2q: dict[int, int] = {}
        checkpoints: list[list[Gate]] = []
        checkpoint_idx: list[int] = []

        for gate in two_qubit_gates:
            qa, qb = gate.qubits
            s = max(cursor_2q.get(qa, -1), cursor_2q.get(qb, -1)) + 1
            while s >= len(checkpoints):
                checkpoints.append([])
            checkpoints[s].append(gate)
            checkpoint_idx.append(s)
            cursor_2q[qa] = cursor_2q[qb] = s

        k = len(checkpoints)

        gap_of_qubit = [0] * num_qubits
        buckets: list[list[Gate]] = [[] for _ in range(k+1)]
        tq_idx = 0
        for gate in gates:
            if gate.type is GateType.TWO_QUBIT:
                qa, qb = gate.qubits
                s = checkpoint_idx[tq_idx]
                gap_of_qubit[qa] = s + 1
                gap_of_qubit[qb] = s + 1
                tq_idx += 1
            else:
                (q,) = gate.qubits
                buckets[gap_of_qubit[q]].append(gate)

        def _pack(bucket: list[Gate]) -> list[list[Gate]]:

            next_slot: dict[int, int] = {}
            substages: list[list[Gate]] = []
            for gate in bucket:
                (q,) = gate.qubits
                slot = next_slot.get(q,0)
                while slot >= len(substages):
                    substages.append([])
                substages[slot].append(gate)
                next_slot[q] = slot + 1
            return substages

        stages: list[list[Gate]] = []
        for i in range(k):
            stages.extend(_pack(buckets[i]))
            stages.append(checkpoints[i])
        stages.extend(_pack(buckets[k]))

        return cls(num_qubits, tuple(tuple(s) for s in stages))

    def two_qubit_gates(self):

        for i, stage in enumerate(self.stages):
            for gate in stage:
                if gate.type is GateType.TWO_QUBIT:
                    yield i, gate

    def interaction_weights(self, delta: float = 0.1) -> dict[tuple[int, int], float]:

        """
        Pairwise interaction weight between logical qubits with decay by stage index
        """

        weights: dict[tuple[int, int], float] = {}
        for stage_idx, gate in self.two_qubit_gates():
            w = math.exp(-delta * stage_idx)
            key = tuple(sorted(gate.qubits))
            weights[key] = weights.get(key, 0.0) + w
        return weights