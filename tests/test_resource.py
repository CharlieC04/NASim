import random
import statistics
from dataclasses import replace
from pathlib import Path

import pytest

from nasim.model import Model
from nasim.resource import Factory, simulate_cultivation

PRESET = Path(__file__).resolve().parents[1] / "src" / "nasim" / "config" / "presets" / "2d.yaml"


@pytest.fixture
def model() -> Model:
    return Model.from_yaml(PRESET)


def test_qec_round_duration(model):
    assert model.qec_round_duration_us == pytest.approx(4 * model.cz_duration_us + model.readout_time_us)


def test_always_succeeds_in_one_attempt(model):
    always = replace(
        model,
        cultivation_check_success_prob=1.0,
        cultivation_escape_success_prob=1.0,
    )
    result = simulate_cultivation(always, random.Random(0))
    assert result.num_check_attempts == 1
    assert result.num_escape_attempts == 1
    assert result.total_duration_us == pytest.approx(
        (always.cultivation_check_cycles + always.cultivation_escape_cycles) * always.qec_round_duration_us
    )


def test_escape_failure_forces_a_fresh_check_attempt(model):
    # Escape never succeeds: every escape attempt is followed by another
    # check attempt, so num_check_attempts tracks num_escape_attempts + 1
    # once the (guaranteed) first check succeeds.
    never_escapes = replace(
        model,
        cultivation_check_success_prob=1.0,
        cultivation_escape_success_prob=0.0,
    )
    rng = random.Random(0)

    class _BoundedRandom(random.Random):
        def __init__(self, seed, limit):
            super().__init__(seed)
            self.calls = 0
            self.limit = limit

        def random(self):
            self.calls += 1
            if self.calls > self.limit:
                raise StopIteration
            return super().random()

    bounded = _BoundedRandom(0, limit=20)
    with pytest.raises(StopIteration):
        simulate_cultivation(never_escapes, bounded)
    # 10 full check+escape pairs consumed (20 calls), then the 11th
    # iteration's check call is the one that trips the limit.
    assert bounded.calls == 21


def test_retry_statistics_match_geometric_distribution(model):
    rng = random.Random(42)
    n = 4000
    results = [simulate_cultivation(model, rng) for _ in range(n)]

    p_check = model.cultivation_check_success_prob
    p_escape = model.cultivation_escape_success_prob
    expected_check_attempts = 1 / (p_check * p_escape)
    expected_escape_attempts = p_check / (p_check * p_escape)

    mean_check = statistics.mean(r.num_check_attempts for r in results)
    mean_escape = statistics.mean(r.num_escape_attempts for r in results)

    assert mean_check == pytest.approx(expected_check_attempts, rel=0.1)
    assert mean_escape == pytest.approx(expected_escape_attempts, rel=0.1)

    for r in results:
        expected_duration = (
            r.num_check_attempts * model.cultivation_check_cycles
            + r.num_escape_attempts * model.cultivation_escape_cycles
        ) * model.qec_round_duration_us
        assert r.total_duration_us == pytest.approx(expected_duration)


def test_deterministic_given_seed(model):
    a = simulate_cultivation(model, random.Random(7))
    b = simulate_cultivation(model, random.Random(7))
    assert a == b


def test_factory_check_ready_us_is_a_fixed_duration(model):
    # FT-Weave's own stage-1 attempt is a fixed duration per restart
    # (success/failure is Bernoulli, not a retry-until-success loop).
    factory = Factory(resource_id=-1, model=model, rng=random.Random(0), distance=3)
    expected = model.cultivation_check_cycles * model.qec_round_duration_us
    assert factory.check_ready_us == pytest.approx(expected)


def test_factory_defaults_num_subfactories_from_the_model(model):
    factory = Factory(resource_id=-1, model=model, rng=random.Random(0), distance=3)
    assert factory.num_subfactories == model.cultivation_num_subfactories
    assert len(factory.subfactories) == model.cultivation_num_subfactories


def test_factory_restart_is_relative_to_now_us_and_clears_subfactories(model):
    factory = Factory(resource_id=-1, model=model, rng=random.Random(0), distance=3, num_subfactories=2)
    factory.simulate_check_stage()
    factory.restart(now_us=10_000.0)
    assert factory.check_ready_us > 10_000.0
    assert all(sf.stage_1_success is None for sf in factory.subfactories)


def test_factory_passed_check_true_iff_any_subfactory_succeeds(model):
    always = replace(model, cultivation_check_success_prob=1.0)
    never = replace(model, cultivation_check_success_prob=0.0)

    passing = Factory(resource_id=-1, model=always, rng=random.Random(0), distance=3, num_subfactories=3)
    passing.simulate_check_stage()
    assert passing.passed_check()

    failing = Factory(resource_id=-2, model=never, rng=random.Random(0), distance=3, num_subfactories=3)
    failing.simulate_check_stage()
    assert not failing.passed_check()


def test_factory_num_spare_successes_excludes_its_own_need(model):
    always = replace(model, cultivation_check_success_prob=1.0)
    factory = Factory(resource_id=-1, model=always, rng=random.Random(0), distance=3, num_subfactories=3)
    factory.simulate_check_stage()
    # 3 successes: 1 kept for its own escape attempt, 2 spare to donate.
    assert factory.num_spare_successes() == 2


def test_factory_consume_and_receive_donated_success(model):
    never = replace(model, cultivation_check_success_prob=0.0)
    factory = Factory(resource_id=-1, model=never, rng=random.Random(0), distance=3, num_subfactories=2)
    factory.simulate_check_stage()
    assert not factory.passed_check()

    factory.receive_donated_success()
    assert factory.passed_check()

    factory.consume_one_success()
    assert not factory.passed_check()

    with pytest.raises(ValueError):
        factory.consume_one_success()


def test_factory_attempt_escape_is_deterministic_given_seed(model):
    a = Factory(resource_id=-1, model=model, rng=random.Random(3), distance=3)
    b = Factory(resource_id=-1, model=model, rng=random.Random(3), distance=3)
    assert a.attempt_escape() == b.attempt_escape()


def test_factory_attempt_escape_duration_matches_model(model):
    factory = Factory(resource_id=-1, model=model, rng=random.Random(0), distance=3)
    _, duration_us = factory.attempt_escape()
    assert duration_us == pytest.approx(model.cultivation_escape_cycles * model.qec_round_duration_us)


def test_factory_attempt_escape_always_succeeds_when_prob_is_one(model):
    always = replace(model, cultivation_escape_success_prob=1.0)
    factory = Factory(resource_id=-1, model=always, rng=random.Random(0), distance=3)
    success, _ = factory.attempt_escape()
    assert success
