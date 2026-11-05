"""Behavioral checks for reviewer-requested isolation of mechanisms."""
import numpy as np
from src.physical_policy import LiveNode, PhysicalPolicy
from src.scheduler_core import operating_penalty, candidate_score
from src.run_r2_simulations import formal_tasks
from pathlib import Path

def node():
    return LiveNode(1, {str(b): b*.1 for b in [1,4,16,64]}, 2., 3., 1000., .001, .01, 'synthetic unit fixture', 40.)

def job():
    return dict(id=1, tenant=0, batch=16, release_s=0., payload_bytes=4224, deadline_s=2., forecast=.5)

def test_matched_baselines_reject_identical_unfunded_request():
    for method in ['compute_only', 'energy_greedy', 'dpp_matched', 'scd']:
        p = PhysicalPolicy([node()], method, budget_factor=.0001, constraint_mode='matched')
        assert p.guard and p.request_guard
        assert p.decide(job(), 0.)['selected'] is None

def test_removing_energy_guard_does_not_remove_request_screening():
    p = PhysicalPolicy([node()], 'scd_no_guard', constraint_mode='matched')
    assert not p.guard and p.request_guard
    p.screened[0] = 4096
    assert p.decide(job(), 0.)['reason'] == 'request_ceiling'

def test_elastic_admission_borrows_but_preserves_device_envelope():
    fixed = PhysicalPolicy([node()], 'scd', budget_factor=.1, constraint_mode='matched')
    elastic = PhysicalPolicy([node()], 'scd_elastic_energy', budget_factor=.1, constraint_mode='matched')
    for p in [fixed, elastic]:
        p.tenant_credit[0,0] = 0.
        p.tenant_debt[0,0] = p.budget[0]/2
    assert fixed.decide(job(), 0.)['selected'] is None
    decision = elastic.decide(job(), 0.)
    assert decision['selected'] == 0
    elastic.commit(job(), decision)
    assert elastic.tenant_credit[0,0] < 0
    assert elastic.credit[0] >= 0
    assert np.isclose(elastic.credit[0]+elastic.debt[0], 2*elastic.budget[0])
    elastic.advance(5.)
    assert np.isclose(elastic.credit[0]+elastic.debt[0], 2*elastic.budget[0])

def test_full_debt_removal_removes_both_feedback_routes():
    args = [1., 1., 2., 1., .1, .2, .5, .1, .3, 1., .8, .9, 0., 40., 65., 15]
    first = operating_penalty(*args)
    args[12] = 4.
    assert operating_penalty(*args) == first
    assert candidate_score(first, 2., 4., 4., 1., 15) == first
    args[-1] = 6
    with_debt = operating_penalty(*args)
    args[12] = 0.
    assert with_debt > operating_penalty(*args)

def test_r2_factorial_is_complete_and_uses_no_request_truncation():
    tasks = formal_tasks(Path('unused'))
    assert len(tasks) == 200
    assert len({t['key'] for t in tasks}) == 200
    assert all(t['scenario']['max_requests'] == 0 for t in tasks)
    assert all((t['scenario']['start_day'], t['scenario']['end_day_exclusive']) == (10,14) for t in tasks)
