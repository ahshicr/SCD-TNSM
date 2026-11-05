"""Reconstruct live commitments and check a same-state shadow drift certificate.

The shadow action minimizes predicted latency on the logged SCD state. It is
not an independently executed policy trajectory. No own-trajectory optimality
claim or unverified conditional-mean energy guarantee follows from this audit.
"""
from __future__ import annotations
import numpy as np
from src.theory_certificate import step_certificate


def audit_trace(events, parameters, spec):
    budgets=np.asarray(parameters['budget_j_per_minute'],dtype=float)
    count=len(budgets)
    tenant_budgets=np.repeat((budgets/4.)[:,None],4,axis=1)
    all_budgets=np.concatenate([budgets,tenant_budgets.reshape(-1)])
    weights=np.concatenate([2.4/budgets,.9/tenant_budgets.reshape(-1)])
    rates=all_budgets/60.
    q=np.zeros(5*count)
    last=0.
    max_state_error=0.
    max_score_error=0.
    max_credit_error=0.
    opportunities=[]
    states=[]
    guarded=spec['method'] not in ['compute_only','energy_greedy','scd_no_guard']
    for event in events:
        kind=event['event']
        stamp=(event['result']['observed_by_policy_s'] if kind=='result' else event.get('observed_s'))
        if stamp is None:
            continue
        if stamp < last-1e-9:
            raise ValueError('Noncausal log order')
        q=np.maximum(q-rates*(stamp-last),0.)
        last=stamp
        if kind=='state':
            observed=np.concatenate([event['debt_j'],np.asarray(event['tenant_debt_j']).reshape(-1)])
            max_state_error=max(max_state_error,float(np.max(np.abs(observed-q))))
            states.append((stamp,observed.copy()))
            if guarded:
                credit=np.concatenate([event['credit_j'],np.asarray(event['tenant_credit_j']).reshape(-1)])
                max_credit_error=max(max_credit_error,float(np.max(np.abs(observed+credit-2*all_budgets))))
        if kind!='decision':
            continue
        decision=event['decision']
        candidates=decision.get('candidates')
        if candidates:
            tenant=event['job']['tenant']
            max_state_error=max(max_state_error,float(np.max(np.abs(
                np.asarray([c['device_debt_j'] for c in candidates])-q[:count]))))
            max_state_error=max(max_state_error,float(np.max(np.abs(
                np.asarray([c['tenant_debt_j'] for c in candidates])-q[count:].reshape(count,4)[:,tenant]))))
        chosen=decision['selected']
        if chosen is None or 'submission_error' in event:
            continue
        tenant=event['job']['tenant']
        charges=np.zeros((count,len(q)))
        for i,candidate in enumerate(candidates):
            charges[i,i]=candidate['predicted_energy_j']
            charges[i,count+i*4+tenant]=candidate['predicted_energy_j']
        if spec['method']=='scd':
            h=np.array([c['operating_penalty'] for c in candidates])
            score=spec['v_joules']*h+charges@(weights*q)
            max_score_error=max(max_score_error,float(np.max(np.abs(score-
                                np.array([c['score'] for c in candidates])))))
            eligible=np.array([c['eligible'] for c in candidates],dtype=bool)
            feasible=np.array([c['feasible'] for c in candidates],dtype=bool)
            filtered=eligible&feasible if (eligible&feasible).any() else eligible
            if not filtered.any():
                raise ValueError('A logged dispatch has no eligible action')
            exact=int(np.where(filtered,score,np.inf).argmin())
            if chosen!=exact:
                raise ValueError('Live SCD action is not the exact filtered score minimum')
            reference=int(np.array([c['predicted_latency_s'] for c in candidates]).argmin())
            opportunities.append(dict(stamp=stamp,q=q.copy(),charges=charges,h=h,
                filtered=filtered,reference=reference,chosen=chosen,identity=event['job']['id']))
        q=q+charges[chosen]
    if max_state_error>1e-6 or max_score_error>1e-6 or max_credit_error>1e-6:
        raise ValueError('Live debt/score/credit reconstruction failed')
    result=dict(max_state_error_j=max_state_error,max_score_error=max_score_error,
                max_guard_credit_invariant_error_j=max_credit_error,
                certificate_scope='Same-state minimum-predicted-latency shadow; NOT an independently executed reference policy')
    if not opportunities:
        result['certificate_opportunities']=0
        return result
    endpoint=next(((stamp,value) for stamp,value in states if stamp>opportunities[-1]['stamp']),None)
    if endpoint is None:
        raise ValueError('Missing logged checkpoint after final admitted action')
    sum_k=sum_g=sum_gamma=sum_difference=0.
    max_residual=-np.inf
    max_transition=0.
    binding=0
    for i,item in enumerate(opportunities):
        next_stamp,next_q=((opportunities[i+1]['stamp'],opportunities[i+1]['q'])
                            if i+1<len(opportunities) else endpoint)
        credits=rates*(next_stamp-item['stamp'])
        certificate=step_certificate(item['q'],weights,item['charges'],credits,item['h'],
            item['filtered'],spec['v_joules'],item['reference'],item['h'][item['reference']])
        if certificate['chosen']!=item['chosen']:
            raise ValueError('Certificate selected another action')
        max_transition=max(max_transition,float(np.max(np.abs(certificate['next_q']-next_q))))
        max_residual=max(max_residual,float(certificate['lhs']-certificate['rhs']))
        sum_k+=certificate['quadratic']
        sum_g+=certificate['resource']
        sum_gamma+=certificate['gate_gap']
        sum_difference+=item['h'][item['chosen']]-item['h'][item['reference']]
        binding+=int(certificate['gate_gap']>1e-9)
    initial_potential=.5*np.dot(weights,opportunities[0]['q']**2)
    final_potential=.5*np.dot(weights,endpoint[1]**2)
    n=len(opportunities)
    bound=(initial_potential-final_potential+sum_k+sum_g+sum_gamma)/(spec['v_joules']*n)
    actual=sum_difference/n
    if max_transition>1e-6 or max_residual>1e-6 or actual>bound+1e-6:
        raise ValueError('Pathwise live certificate failed')
    result.update(certificate_opportunities=n,quadratic_sum_j=sum_k,resource_sum_j=sum_g,
        restriction_gap_sum_j=sum_gamma,restriction_binding_fraction=binding/n,
        mean_shadow_cost_difference=actual,finite_horizon_upper_bound=bound,
        maximum_step_inequality_residual=max_residual,max_queue_transition_error_j=max_transition,
        certificate_end_s=endpoint[0],state_transfer_term=0.,
        state_transfer_explanation='Zero by definition for the declared same-state shadow, not asserted for own-policy trajectories')
    return result
