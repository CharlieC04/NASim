from __future__ import annotations

from dataclasses import dataclass

# using qSIEVE table 4
LOAD_LOGICAL_CYCLES = 2
STORE_LOGICAL_CYCLES = 1

def load_duration_us(qec_round_duration_us: float, code_distance: int) -> float:
    return LOAD_LOGICAL_CYCLES * code_distance * qec_round_duration_us

def store_duration_us(qec_round_duration_us: float, code_distance: int) -> float:
    return STORE_LOGICAL_CYCLES * code_distance * qec_round_duration_us

@dataclass
class LoadStoreAncilla:

    """
    LD/ST interface between one qLDPC memory block and surface code compute region.
    Asm memory block can serve one load or store at a time
    """

    memory_block_id: int
    busy_until_us: float = 0.0
    in_flight_qubit: int | None = None

    def is_free(self, now_us: float) -> bool:
        return now_us >= self.busy_until_us

    def begin(self, *, qubit: int, now_us: float, duration_us: float) -> None:

        if not self.is_free(now_us):
            raise ValueError('finish called before op done')

        self.in_flight_qubit = qubit
        self.busy_until_us = now_us + duration_us

    def finish(self, now_us: float):
        if now_us < self.busy_until_us:
            raise ValueError('finish called before op done')

        self.in_flight_qubit = None