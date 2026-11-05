import numpy as np
import pytest
from src.theory_certificate import step_certificate


def test_pathwise_certificate_all_reference_actions_random_states():
    rng = np.random.default_rng(93271)
    for _ in range(1000):
        q = rng.uniform(0, 100, 5)
        w = rng.uniform(.01, 10, 5)
        a = rng.uniform(0, 3, (4, 5))
        b = rng.uniform(0, 5, 5)
        h = rng.uniform(0, 123.002025, 4)
        feasible = rng.integers(0, 2, 4).astype(bool)
        v = float(np.exp(rng.uniform(-5, 5)))
        for reference in range(4):
            c = step_certificate(q, w, a, b, h, feasible, v, reference,
                                 float(rng.uniform(0, 123.002025)))
            assert c['lhs'] <= c['rhs']+1e-8
            assert c['gate_gap'] >= -1e-8


def test_gate_exact_scan_does_not_imply_zero_unrestricted_gap():
    c = step_certificate([0], [1], [[0], [0]], [0], [0, 2], [False, True], 8., 0, 0.)
    assert c['chosen'] == 1
    assert c['gate_gap'] == 16.  # scales with V despite exact candidate scan
    assert c['lhs'] == c['rhs'] == 16.


def test_independent_trajectory_cost_cannot_replace_state_cost():
    # State X: greedy cost 0 -> absorbing bad state (cost 1 each step).
    # Alternative cost 2 -> absorbing good state (cost 0 each step).
    # Both have strictly feasible resource increments and bounded cost.
    # Hence feasibility alone cannot justify an O(1/V) own-trajectory bound.
    n = 100
    greedy_mean, other_mean = (n-1)/n, 2/n
    v, c0 = 100., .5*(.1-1.)**2
    assert greedy_mean > other_mean+c0/v
    c = step_certificate([0], [1], [[.1], [.1]], [1], [1, 1], [True, True],
                         v, 1, 0.)
    assert c['state_transfer'] == 1.
    assert c['lhs'] <= c['rhs']


def test_telescoped_finite_horizon_certificate():
    rng = np.random.default_rng(69)
    q, w, v = np.zeros(3), np.array([.4, .5, .9]), 2.
    accumulated_lhs = accumulated_rhs = 0.
    for _ in range(300):
        c = step_certificate(q, w, rng.uniform(0, 2, (3, 3)), rng.uniform(0, 1, 3),
                             rng.uniform(0, 4, 3), rng.integers(0, 2, 3).astype(bool),
                             v, int(rng.integers(3)), float(rng.uniform(0, 4)))
        q = c['next_q']
        accumulated_lhs += c['lhs']
        accumulated_rhs += c['rhs']
    assert accumulated_lhs <= accumulated_rhs+1e-8
