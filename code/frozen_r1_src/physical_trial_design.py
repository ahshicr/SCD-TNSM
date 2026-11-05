"""Frozen matched-block live experiment design, separate from online decisions.

The generator may read the entire chosen trace window to create offered work.
The controller receives one released job at a time, with a past-only forecast.
Azure timing/tenant/class proportions drive a *new* CPU-MLP workload, not a claim
that the lost applications or accelerator kernels have been reconstructed.
"""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import numpy as np
from src.physical_policy import LiveNode


BASE_METHODS = ['compute_only', 'energy_greedy', 'dpp_matched', 'scd']
COHORTS = {'fleet': [1, 3, 5, 6, 7, 8, 9, 10, 12, 17, 19, 20, 21, 22, 23],
           'poe': [5, 6, 7, 8, 9]}
SCENARIOS = ['nominal', 'network_pressure', 'energy_pressure']


def design(repetitions=10):
    if repetitions < 10:
        raise ValueError('Formal protocol requires at least ten matched blocks')
    trials = []
    for repeat in range(repetitions):
        rng = random.Random(20260908+repeat)
        cells = [(cohort, scenario) for cohort in COHORTS for scenario in SCENARIOS]
        rng.shuffle(cells)
        for cohort, scenario in cells:
            methods = BASE_METHODS[:]
            if cohort == 'poe' and scenario == 'network_pressure':
                methods += ['scd_no_link']
            if cohort == 'poe' and scenario == 'energy_pressure':
                methods += ['scd_no_debt', 'scd_no_guard']
            if cohort == 'poe' and scenario == 'nominal':
                methods += ['scd_no_thermal']
            rng.shuffle(methods)
            for method in methods:
                trials.append(dict(ordinal=len(trials), block=repeat, cohort=cohort,
                    scenario=scenario, method=method, duration_s=60., drain_s=15., idle_s=20.,
                    jobs=2880 if cohort == 'fleet' else 720,
                    budget_factor=.06 if scenario=='energy_pressure' else .5,
                    v_joules=1., energy_margin=1.1,
                    trace_start=2880+1440*(repeat % 12)+300+3*(repeat//12),
                    trace_minutes=60, job_seed=20260909+1009*repeat,
                    hardware_guard_c=65., max_inflight_per_board=4))
    return trials


def jobs_for_trial(spec, aggregate, forecast):
    start, width = spec['trace_start'], spec['trace_minutes']
    counts = aggregate['count'][start:start+width]
    if counts.shape != (width, 16, 4) or np.any(counts < 0) or counts.sum() <= 0:
        raise ValueError('Invalid trace window')
    rng = np.random.default_rng(spec['job_seed'])
    indices = rng.choice(counts.size, size=spec['jobs'], p=counts.reshape(-1)/counts.sum())
    minutes, classes, tenants = np.unravel_index(indices, counts.shape)
    releases = (minutes+rng.uniform(.05, .95, len(indices)))*spec['duration_s']/width
    jobs = []
    for ordinal in np.argsort(releases, kind='stable'):
        minute, cls, tenant = int(minutes[ordinal]), int(classes[ordinal]), int(tenants[ordinal])
        batch = [1, 4, 16, 16][cls//4]
        payloads = [batch*66*4, 65536, 65536, 1048576]
        if spec['scenario'] == 'network_pressure':
            payloads = [65536, 1048576, 1048576, 4194304]
        trace_minute = start+minute
        origin = int(forecast['model_origin'][trace_minute])
        if origin > trace_minute:
            raise ValueError('Forecast model originates in the future')
        pressure = forecast['prediction'][trace_minute, cls]/max(
            forecast['normalization_scale'][trace_minute, cls], 1.)
        jobs.append(dict(id=len(jobs), release_s=float(releases[ordinal]), tenant=tenant,
            batch=batch, payload_bytes=max(payloads[cls%4], batch*66*4),
            deadline_s={1:.15, 4:.30, 16:.75}[batch], class_id=cls,
            trace_minute=trace_minute, forecast_origin=origin,
            forecast=float(np.clip(pressure, 0., 3.)),
            feature_start=int((ordinal*7+cls*3)%128)))
    return jobs


def build_nodes(cohort, service, transport, power, initial):
    services = {r['port']: r for r in service['boards'] if r['complete']}
    links = {r['port']: r['controller_transport_fit'] for r in transport['transport'] if r['complete']}
    calibrated = {r['port']: r for r in power['boards']
                  if (r.get('achieved_occupancy_fit') or {}).get('repeated_active_calibration')}
    if set(calibrated) != set(COHORTS['poe']):
        raise ValueError('Unexpected complete active-power calibration cohort')
    nodes = []
    for port in COHORTS[cohort]:
        if port not in services or port not in links:
            raise ValueError('Missing completed service/isolated transport calibration for port %s' % port)
        # Cross-device transfers are explicit modeling assumptions, never reported
        # as individual measured whole-board power. Unmetered non-Pi types have
        # declared scenario coefficients; the five-port primary energy trial
        # does not use these coefficients or any mixed measurement boundary.
        reference = port if port in calibrated else (5 if port in [1, 3, 19] else
                    9 if port in [10, 17, 21, 22] else 6 if port == 23 else None)
        if reference is not None:
            fit = calibrated[reference]['achieved_occupancy_fit']
            idle, incremental = fit['idle_intercept_w'], fit['active_increment_w']
            provenance = ('Repeated individual PSE calibration' if reference == port else
                          'Same-family transfer from port %s; not individual power measurement' % reference)
        else:
            idle, incremental = {12: (4., 3.), 20: (4., 4.)}[port]
            provenance = 'Declared unmetered platform scenario; not a measured coefficient'
        telemetry = initial[str(port)].get('telemetry', {})
        temperature = telemetry.get('cpu_c')
        healthy = (temperature is not None and np.isfinite(temperature) and temperature < 65.
                   and not initial[str(port)].get('thermal_halted', False))
        nodes.append(LiveNode(port, {b: r['service_mean_s'] for b, r in services[port]['batch_models'].items()},
            idle, incremental, 100. if port in [6, 7, 8] else 1000.,
            links[port]['fixed_overhead_s'], links[port]['seconds_per_decimal_mb'],
            provenance, temperature, healthy=bool(healthy)))
    return nodes


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def source_hashes():
    names = ['physical_trial_design.py', 'run_physical_trials.py', 'physical_policy.py',
             'scheduler_core.py', 'board_pipeline.py', 'board_inference_client.py',
             'board_inference_worker.py', 'portable_inference.py', 'calibrate_power_thermal.py',
             'poe_sampling_audit.py', 'switch_transport.py']
    return {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names}
