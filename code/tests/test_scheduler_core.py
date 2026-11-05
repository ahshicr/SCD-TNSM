import math

import numpy as np
import pytest

from src.scheduler_core import (
    admission_tokens, update_demand, path_timing, hop_jitter_mean,
    operating_penalty, candidate_score, select_candidate,
    legitimate_success, OPERATING_PENALTY_BOUND,
    deposit_execution_heat, observed_free_time,
    refill_energy_credit, energy_eligible,
)


def test_admission_ceiling_and_integer_requests():
    for previous in (0, 1, 64, 256, 1e12):
        for budget in (.01, .1, .3, .5, 1, 10):
            result = admission_tokens(previous, 256, budget)
            assert isinstance(result, int)
            assert 0 <= result <= 256
    assert admission_tokens(64, 256, 1) == 70


def test_attack_can_poison_only_its_own_bounded_history():
    before = np.array([4., 16., 64., 24.])
    clean = np.array([6., 18., 60., 22.])
    attack = clean.copy()
    attack[-1] += 1e10
    a = [update_demand(x, y, .2, 256) for x, y in zip(before, clean)]
    b = [update_demand(x, y, .2, 256) for x, y in zip(before, attack)]
    assert a[:3] == b[:3]
    assert b[-1] == 256
    assert admission_tokens(b[-1], 256, 1) == 256


def test_hop_jitter_is_once_per_hop_and_blocks_following_work():
    a = path_timing(0., 0., 0., 0., 1., 0., 1., .2, 0., .3)
    assert a == pytest.approx((1.2, 1.2, 2.5, 0.))
    b = path_timing(0., a[0], 0., a[2], 1., 0., 1., 0., 0., 0.)
    assert b == pytest.approx((2.2, 2.2, 3.5, 1.5))
    c = path_timing(0., 0., 5., 0., 1., 2., 1., .2, .1, .3)
    assert c == pytest.approx((1.2, 7.1, 8.4, 3.8))
    assert hop_jitter_mean(0., 0.) == .00005


def test_switch_and_compute_can_overlap_between_jobs():
    a = path_timing(0., 0., 0., 0., .1, 0., .1, 0., 0., 0.)
    compute_end = a[2]+10
    b = path_timing(.3, a[0], 0., a[2], .1, 0., .1, 0., 0., 0.)
    assert b[2] < compute_end  # second transfer need not wait for first compute


def test_unobserved_future_completion_cannot_change_reservation():
    assert observed_free_time(10., 8., 4.) == observed_free_time(10., 800., 4.) == 10.
    assert observed_free_time(10., 3., 4.) == 4.  # completed event is observable
    assert observed_free_time(2., 800., 4.) == 4.  # overdue, but no future oracle


def test_heat_is_deposited_when_executed_not_when_dispatched():
    calendar = np.zeros((4, 2))
    deposit_execution_heat(calendar, 1, 90., 150., 4.)
    assert calendar[:, 1] == pytest.approx([0., 120., 120., 0.])
    assert calendar[:, 0].sum() == 0
    assert calendar.sum() == 240


def test_scd_dpp_identical_score_and_gate_only_difference():
    rng = np.random.default_rng(811)
    for _ in range(2000):
        v = float(np.exp(rng.uniform(-6, 6)))
        params = (rng.uniform(0, 100), rng.uniform(.1, 4),
                  rng.uniform(.01, 100), .01, rng.uniform(0, 20),
                  rng.uniform(0, 10), rng.uniform(.001, 10), rng.uniform(0, 10),
                  rng.uniform(0, 100), rng.uniform(.01, 10),
                  rng.uniform(0, 1000), rng.uniform(0, 1000),
                  rng.uniform(0, 1000), rng.uniform(10, 200), 72.)
        h5 = operating_penalty(*params, 5)
        h10 = operating_penalty(*params, 10)
        assert 0 <= h5 == h10 <= OPERATING_PENALTY_BOUND
        e, q, z = rng.uniform(0, 100, 3)
        assert candidate_score(h5, e, q, z, v, 5) == candidate_score(h10, e, q, z, v, 10)
    costs = np.array([1., 2., 3.])
    feasible = np.array([False, True, True])
    assert select_candidate(costs, feasible, False) == 0
    assert select_candidate(costs, feasible, True) == 1
    assert select_candidate(costs, np.zeros(3, dtype=bool), True) == 0
    assert select_candidate(np.ones(3), np.ones(3, dtype=bool), True) == 0


def test_no_debt_ablation_removes_device_and_tenant_debt_terms() -> None:
    h, energy, device_debt, tenant_debt, v = 2.0, 3.0, 4.0, 5.0, 1.5
    assert candidate_score(h, energy, device_debt, tenant_debt, v, 6) == v * h
    assert candidate_score(h, energy, device_debt, tenant_debt, v, 13) == (
        v * h + 2.40 * device_debt * energy
    )


def test_normal_success_denominator_includes_drops_excludes_only_attacks():
    assert legitimate_success(80, 150, 50) == .8
    assert legitimate_success(0, 0, 0) == 0


def test_credit_and_debt_invariant_without_feasible_arrivals():
    rng = np.random.default_rng(47)
    rate, capacity = 3., 6.
    credit, debt, time_s, charged = capacity, 0., 0., 0.
    for _ in range(20000):
        elapsed = float(rng.exponential(.1))
        time_s += elapsed
        credit = refill_energy_credit(credit, rate, elapsed, 2.)
        debt = max(0., debt-rate*elapsed/60.)
        cost = float(rng.lognormal(2, 2))  # intentionally grossly infeasible demand
        if energy_eligible(cost, credit, credit, True):
            credit -= cost
            debt += cost
            charged += cost
        assert credit >= 0 and debt <= capacity+1e-10
        assert credit+debt == pytest.approx(capacity, abs=1e-9)
        assert charged <= capacity+rate*time_s/60.+1e-8
