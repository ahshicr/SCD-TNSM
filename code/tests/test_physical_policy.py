from copy import deepcopy
import pytest
from src.physical_policy import LiveNode, PhysicalPolicy


def nodes():
    return [LiveNode(i+1, {str(b): b*s for b in [1, 4, 16, 64]}, 2., watts,
                     1000., .001, .01, 'test fixture, not measurements', 40.)
            for i, (s, watts) in enumerate([(.01, 3.), (.03, .5), (.1, 1.)])]


def job(identity=1):
    return dict(id=identity, tenant=0, batch=16, release_s=0., payload_bytes=4224,
                deadline_s=.2, forecast=.5)


def test_live_scd_and_dpp_share_scores_and_differ_only_in_gate():
    scd = PhysicalPolicy(nodes(), 'scd', guard=False)
    dpp = PhysicalPolicy(nodes(), 'dpp_matched', guard=False)
    scd.debt[0] = 4*scd.budget[0]
    dpp.debt[0] = 4*dpp.budget[0]
    a, b = scd.decide(job(), 0.), dpp.decide(job(), 0.)
    assert [c['score'] for c in a['candidates']] == [c['score'] for c in b['candidates']]
    assert [c['operating_penalty'] for c in a['candidates']] == [c['operating_penalty'] for c in b['candidates']]
    assert a['selected'] == 0
    assert b['selected'] == 1


def test_live_credit_envelope_and_hardware_guard_survive_ablation():
    policy = PhysicalPolicy(nodes(), 'scd', budget_factor=.0001)
    result = policy.decide(job(), 0.)
    assert result['selected'] is None
    for node in policy.nodes:
        node.temperature_c = 65.
    no_thermal = PhysicalPolicy(deepcopy(policy.nodes), 'scd_no_thermal')
    assert no_thermal.decide(job(), 0.)['selected'] is None


def test_completion_reconciles_only_observed_time_not_remote_future():
    first, second = PhysicalPolicy(nodes()), PhysicalPolicy(nodes())
    for policy in [first, second]:
        decision = policy.decide(job(), 0.)
        policy.commit(job(), decision)
        assert len(policy.nodes[decision['selected']].pending) == 1
    result = dict(id=1, status='ok', after={'cpu_c': 43.}, compute_end=999999.)
    first.completed(decision['selected'], result, .25)
    result['compute_end'] = -999999.
    second.completed(decision['selected'], result, .25)
    assert first.nodes[decision['selected']].predicted_compute_free == .25
    assert first.nodes == second.nodes
    with pytest.raises(ValueError):
        first.advance(.1)
