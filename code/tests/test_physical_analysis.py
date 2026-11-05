from copy import deepcopy
import numpy as np
import pytest
from src.physical_policy import LiveNode,PhysicalPolicy
from src.physical_trace_audit import audit_trace
from src.summarize_physical_trials import weighted_quantile,integrate_observed,interval
from src.audit_physical_power_sampling import interval_support


def test_weighted_latency_quantile_is_request_not_batch_weighted():
    assert weighted_quantile([.1,1.],[1,16],.5)==1.
    assert weighted_quantile([.1,1.],[1,16],.05)==.1
    with pytest.raises(ValueError):
        weighted_quantile([],[],.95)


def test_power_integration_common_window_and_strict_gap_checks():
    assert integrate_observed([0,1,2,3],[2,2,2,2],.5,2.5)==4.
    assert integrate_observed([0,1,2,3],[0,1,2,3],.5,2.5)==3.
    with pytest.raises(ValueError,match='support'):
        integrate_observed([0,1,2],[2,2,2],-1,2)
    with pytest.raises(ValueError,match='gap'):
        integrate_observed([0,1,5,6],[2,2,2,2],.5,5.5)


def test_power_integration_ignores_only_post_window_reauthentication_gap():
    # The interval itself is sampled through t=2.0; a reconnect delays the
    # post-window point.  Only the 0.2 s terminal tail is held, not the 3 s gap.
    assert integrate_observed([0,1,2,5],[2,2,4,9],.5,2.2)==pytest.approx(4.8)
    assert integrate_observed([0,1,2],[2,2,4],.5,2.2)==pytest.approx(4.8)
    with pytest.raises(ValueError,match='timely'):
        integrate_observed([0,1,2,5],[2,2,4,9],.5,4.8)


def test_power_support_distinguishes_post_window_gap_from_internal_gap():
    post = interval_support(np.asarray([0., 1., 2., 5.]), .5, 2.2)
    assert post['supported']
    assert post['terminal_hold_required']
    assert post['two_sided_bracketing_would_reject']
    assert post['offending_gaps'][0]['location'] == 'crosses_stop'
    internal = interval_support(np.asarray([0., 1., 5., 6.]), .5, 5.5)
    assert not internal['supported']
    assert internal['in_window_offending_gaps'][0]['location'] == 'internal'


def test_intervals_require_all_ten_blocks_and_keep_missing_boundary():
    assert interval([2.]*10)==(2.,0.,10)
    assert interval([None]*10)==(None,None,0)
    with pytest.raises(ValueError):
        interval([1.]*9+[None])


def trace_fixture():
    nodes=[LiveNode(p,{str(b):b*s for b in [1,4,16,64]},2.,1.,1000.,.001,.01,'test-only',40.)
           for p,s in [(5,.01),(9,.02)]]
    policy=PhysicalPolicy(nodes,'scd',budget_factor=.5)
    events=[]
    for identity in range(20):
        now=identity*.2
        job=dict(id=identity,tenant=identity%4,batch=4,release_s=now,payload_bytes=1056,
                 deadline_s=.1,forecast=.5)
        decision=policy.decide(job,now)
        assert decision['selected'] is not None
        events.append(dict(event='decision',job=job,observed_s=now,decision=decision))
        policy.commit(job,decision)
        result=dict(id=identity,status='ok',after={'cpu_c':40.},observed_by_policy_s=now+.1)
        policy.completed(decision['selected'],result,now+.1)
        events.append(dict(event='result',result=result))
        events.append(dict(event='state',observed_s=now+.1,debt_j=policy.debt.tolist(),
            credit_j=policy.credit.tolist(),tenant_debt_j=policy.tenant_debt.tolist(),
            tenant_credit_j=policy.tenant_credit.tolist()))
    return events,dict(budget_j_per_minute=policy.budget.tolist()),dict(method='scd',v_joules=1.)


def test_actual_state_certificate_and_corrupt_log_detection():
    events,parameters,spec=trace_fixture()
    result=audit_trace(events,parameters,spec)
    assert result['certificate_opportunities']==20
    assert result['max_state_error_j']<1e-10
    assert result['mean_shadow_cost_difference']<=result['finite_horizon_upper_bound']+1e-10
    corrupted=deepcopy(events)
    next(e for e in corrupted if e['event']=='state')['debt_j'][0]+=1.
    with pytest.raises(ValueError,match='reconstruction'):
        audit_trace(corrupted,parameters,spec)
