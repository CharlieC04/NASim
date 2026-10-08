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