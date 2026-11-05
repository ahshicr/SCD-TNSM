"""Pathwise drift certificate; does not equate two policy state trajectories."""
from __future__ import annotations
import numpy as np
from src.scheduler_core import select_candidate


def step_certificate(q, weights, charges, credits, h, feasible, v, reference_action,
                     reference_own_h):
    q, weights, charges, credits, h = (np.asarray(x, dtype=float)
                                      for x in (q, weights, charges, credits, h))
    scores = v*h+charges@(weights*q)
    chosen = select_candidate(scores, np.asarray(feasible, dtype=bool), True)
    increment = charges[chosen]-credits
    next_q = np.maximum(q+increment, 0.)
    drift = .5*np.dot(weights, next_q**2-q**2)
    quadratic = .5*np.dot(weights, increment**2)
    gate_gap = scores[chosen]-scores.min()
    resource = np.dot(weights*q, charges[reference_action]-credits)
    state_transfer = h[reference_action]-reference_own_h
    lhs = drift+v*(h[chosen]-reference_own_h)
    rhs = quadratic+resource+gate_gap+v*state_transfer
    return dict(chosen=chosen, next_q=next_q, drift=drift, quadratic=quadratic,
                gate_gap=gate_gap, resource=resource, state_transfer=state_transfer,
                lhs=lhs, rhs=rhs)
