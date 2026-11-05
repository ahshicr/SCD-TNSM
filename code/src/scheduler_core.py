"""Shared, deterministic scheduling primitives (seconds, joules, Celsius).

No primitive accepts an attack label, future traffic, or future completion.
Numba compilation is optional so the same rules can be inspected without it.
"""
from __future__ import annotations

import math
import numpy as np

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        return lambda fn: fn


@njit(cache=True)
def batch_work(per_request_work, requests):
    """Sequential reference inference conserves admitted work; no unmeasured speedup."""
    return per_request_work*requests


@njit(cache=True)
def within_phase_release(minute_start, phase_ordinal, phase_count, chunk, chunks, jitter):
    """Ordered releases stay inside their phase, including batches split at 64."""
    return minute_start+60.*(phase_ordinal+(chunk+.10+.80*jitter)/chunks)/phase_count


@njit(cache=True)
def admission_tokens(demand_ewma, cap, budget_fraction):
    """Previous-window observed demand; immutable administrator ceiling.

    EWMA includes *all* offered requests, including rejected/malicious ones.
    Thus poisoning can raise a tenant's allowance only to its configured cap.
    Tokens are integer requests; no fractional request can be admitted.
    """
    factor = min(1.0, max(0.30, 2.0 * budget_fraction))
    return int(math.floor(min(cap, math.ceil(1.05 * demand_ewma + 2.0)) * factor))


@njit(cache=True)
def update_demand(previous, observed, alpha, cap):
    return min(cap, (1.0-alpha)*previous + alpha*observed)


@njit(cache=True)
def refill_energy_credit(balance, budget_per_minute, elapsed_s, burst_windows):
    return min(budget_per_minute*burst_windows, balance+budget_per_minute*elapsed_s/60.)


@njit(cache=True)
def energy_eligible(charge, device_credit, tenant_credit, isolate):
    return charge <= device_credit and (not isolate or charge <= tenant_credit)


@njit(cache=True)
def hop_jitter_mean(serialization, cross_fraction):
    x = min(max(cross_fraction, 0.0), 0.95)
    return 0.00005 + serialization*(0.015 + 0.32*x*x/max(1.0-x, 0.05))


@njit(cache=True)
def path_timing(release, ingress_free, inter_free, port_free,
                ingress_s, inter_s, port_s, ingress_jitter, inter_jitter, port_jitter):
    """Store-and-forward FCFS path; jitter is extra per-hop service occupancy.

    Absent inter-switch hop has zero serialization and is skipped entirely.
    Queue waiting excludes both serialization and random service occupancy.
    Returns each departure time and the sum of *waiting* over traversed hops.
    """
    t1 = max(release, ingress_free)
    queue = t1-release
    t1 += ingress_s+ingress_jitter
    t2 = t1
    if inter_s > 0.0:
        queue += max(0.0, inter_free-t1)
        t2 = max(t1, inter_free)+inter_s+inter_jitter
    queue += max(0.0, port_free-t2)
    t3 = max(t2, port_free)+port_s+port_jitter
    return t1, t2, t3, queue


@njit(cache=True)
def operating_penalty(latency, deadline, energy, min_energy, compute_wait,
                      network, service, switch_wait, forecast, capacity,
                      current_ratio, projected_ratio, debt_ratio, temperature,
                      thermal_threshold, method):
    """Explicitly bounded non-debt h; every bound is independent of V.

    SCD (10) and matched DPP (5) intentionally take exactly the same branch.
    Clipping changes priority saturation, not actual reporting or debt charges.
    """
    lat = min(max(latency/max(deadline, 1e-9), 0.0), 8.0)
    er = min(max(energy/max(min_energy, 1e-9), 0.0), 8.0)
    q = min(max(compute_wait/max(deadline, 1e-9), 0.0), 8.0)
    swq = min(max(switch_wait/max(deadline, 1e-9), 0.0), 8.0)
    f = min(max(forecast, 0.0), 3.0) if method != 8 else 0.0
    cur = min(max(current_ratio, 0.0), 8.0)
    projected = min(max(projected_ratio, 0.0), 8.0)
    # Method 6 reproduces the R1 explicit-price ablation. Method 15 also
    # removes debt from the dynamic energy weight, retaining hard eligibility.
    debt = min(max(debt_ratio, 0.0), 4.0) if method != 15 else 0.0
    wl, we = 0.88, 0.52
    if method != 9:
        wl *= 1.0+0.20*f
        we *= 1.0+1.40*max(0.0, cur-0.65)+0.55*debt
    h = wl*lat+we*er+0.34*q
    if method != 7:
        link = min(max(network/max(network+service, 1e-9), 0.0), 1.0)
        h += 0.42*link+0.24*swq
    h += 0.13*f/max(capacity, 0.25)
    h += 0.65*max(0.0, projected-0.85)**2
    if method != 11:
        thermal = min(max((temperature-thermal_threshold+8.0)/12.0, 0.0), 8.0)
        h += 0.44*thermal
    return h+1.75*max(0.0, lat-1.0)


OPERATING_PENALTY_BOUND = (
    0.88*1.6*8 + 0.52*(1+1.40*(8-0.65)+0.55*4)*8 + 0.34*8
    + 0.42+0.24*8+0.13*3/0.25+0.65*(8-0.85)**2+0.44*8+1.75*7
)


@njit(cache=True)
def candidate_score(h, energy, debt_ratio, tenant_ratio, v, method):
    score = v*h
    if method != 6 and method != 15:
        score += 2.40*debt_ratio*energy
    if method != 6 and method != 13 and method != 15:
        score += 0.90*tenant_ratio*energy
    return score


@njit(cache=True)
def select_candidate(scores, feasible, gate):
    restrict = gate and np.any(feasible)
    best, best_score = -1, math.inf
    for i in range(len(scores)):
        if restrict and not feasible[i]:
            continue
        if scores[i] < best_score:
            best, best_score = i, scores[i]
    return best


def legitimate_success(on_time, total_offered, attacker_offered):
    """Rejections count as failures; thermal events never alter denominator."""
    return on_time/max(total_offered-attacker_offered, 1.0)


@njit(cache=True)
def deposit_execution_heat(calendar, node, start, stop, watts):
    """Integrate execution power into its actual wall-clock windows.

    Only completed accounting windows may be read by the controller. Energy
    scheduled after the trace's arrival horizon is not a temperature sample.
    """
    if stop <= start:
        return
    first = max(0, int(start//60.0))
    last = min(len(calendar)-1, int(stop//60.0))
    for window in range(first, last+1):
        overlap = max(0.0, min(stop, (window+1)*60.)-max(start, window*60.))
        calendar[window, node] += watts*overlap


@njit(cache=True)
def observed_free_time(predicted, actual_last_completion, now):
    """Completion feedback reveals idle/busy, never a future completion time."""
    if actual_last_completion <= now:
        return now
    return max(now, predicted)
