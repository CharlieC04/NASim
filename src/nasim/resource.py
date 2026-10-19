from __future__ import annotations

from dataclasses import dataclass, field
import random

from nasim.model import Model

@dataclass(frozen=True)
class CultivationResult:

    num_check_attempts: int
    num_escape_attempts: int
    total_duration_us: float

def simulate_cultivation(model: Model, rng: random.Random) -> CultivationResult:

    num_check = num_escape = 0
    total_us = 0.0
    while True:
        num_check += 1
        total_us += model.cultivation_check_cycles * model.qec_round_duration_us
        if rng.random() >= model.cultivation_check_success_prob: continue

        num_escape += 1
        total_us += model.cultivation_escape_cycles * model.qec_round_duration_us
        if rng.random() < model.cultivation_escape_success_prob:
            return CultivationResult(num_check, num_escape, total_us)

@dataclass
class SubFactory:

    index: int
    stage_1_success: bool | None = None

@dataclass
class Factory:

    """
    magic state factory based on FT-weave. It has num_subfactories parallel check-stage lines
    """

    resource_id: int
    model: Model
    rng: random.Random
    distance: int
    num_subfactories: int | None = None
    subfactories: list[SubFactory] = field(init=False)
    check_ready_us: float = field(init=False)

    def __post_init__(self):

        if self.num_subfactories is None:
            self.num_subfactories = self.model.cultivation_num_subfactories

        self.subfactories = [SubFactory(index=i) for i in range(self.num_subfactories)]
        self.check_ready_us = self.model.cultivation_check_cycles * self.model.qec_round_duration_us

    def restart(self, now_us: float):

        for sf in self.subfactories:
            sf.stage_1_success = None

        self.check_ready_us = now_us + self.model.cultivation_check_cycles * self.model.qec_round_duration_us

    def simulate_check_stage(self):

        for sf in self.subfactories:
            sf.stage_1_success = self.rng.random() < self.model.cultivation_check_success_prob

    def passed_check(self) -> bool:

        return any(sf.stage_1_success for sf in self.subfactories)

    def num_spare_successes(self) -> int:

        return max(0, sum(1 for sf in self.subfactories if sf.stage_1_success) - 1)

    def consume_one_success(self):

        for sf in self.subfactories:
            if sf.stage_1_success:
                sf.stage_1_success = False
                return

        raise ValueError("Factory has no success")

    def receive_donated_success(self):

        """
        Can donate spare success check to other factory
        """

        for sf in self.subfactories:
            if not sf.stage_1_success:
                sf.stage_1_success = True
                return

        raise ValueError("Factory has no free slot")

    def attempt_escape(self) -> tuple[bool, float]:

        duration_us = self.model.cultivation_escape_cycles * self.model.qec_round_duration_us
        success = self.rng.random() < self.model.cultivation_escape_success_prob
        return success, duration_us