"""Full matched-block, real-MLP scheduler trials with local append-only logs.

Nothing is installed globally, no switch configuration is changed, and the
65 C per-inference safety guard remains active for every policy/ablation.
Passwords are read interactively into memory, never into a manifest or argv.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import getpass
import hashlib
import json
import os
from pathlib import Path
import queue
import time
import numpy as np
from src.board_inference_client import BoardClient
from src.board_pipeline import BoardPipeline
from src.calibrate_power_thermal import PowerSampler
from src.physical_policy import PhysicalPolicy
from src.physical_trial_design import (COHORTS, design, jobs_for_trial, build_nodes,
                                       canonical_hash, source_hashes)


def utc():
    return datetime.now(timezone.utc).isoformat()


def dump(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def record(handle, value):
    handle.write(json.dumps(value, separators=(',', ':'), allow_nan=False)+'\n')
    handle.flush()


def check_sources(expected):
    if source_hashes() != expected:
        raise RuntimeError('A frozen live source changed; refusing a mixed-code campaign')


def select_service_batch(examples, job):
    """Keep each released service's inputs and parity targets paired."""
    if 'services' in examples:
        data = examples['services'][job.get('service_id', 'mlp')]
    else:
        data = examples
    indices = (np.arange(job['batch'])+job['feature_start']) % len(data['features'])
    return data['features'][indices], data['expected'][indices]


def run_trial(spec, directory, rows, key, model, examples, datasets, sampler,
              job_generator=jobs_for_trial, node_builder=build_nodes):
    directory.mkdir(exist_ok=False)
    jobs = job_generator(spec, datasets['aggregate'], datasets['forecast'])
    dump(directory/'offered_jobs.json', jobs)
    protocol = dict(**spec, started_utc=utc(), offered_sha256=canonical_hash(jobs),
                    offered_batches=len(jobs), offered_inferences=sum(j['batch'] for j in jobs))
    dump(directory/'protocol.json', protocol)
    clients, pipelines, ready, initial = {}, {}, {}, {}
    summary = dict(complete=False, protocol=protocol, error=None, successful_batches=0,
                   successful_inferences=0, returned_individual_predictions=0,
                   rejected_batches=0, dispatched_batches=0, guarded_batches=0,
                   parity_failures=0, failed_batches=0, cleanup={})
    try:
        def initialize(row):
            return BoardClient(row, key, model)
        with ThreadPoolExecutor(max_workers=len(rows)) as pool:
            futures = {pool.submit(initialize, row): row['port'] for row in rows}
            for future in as_completed(futures):
                port = futures[future]
                try:
                    clients[port] = future.result()
                    ready[str(port)] = clients[port].ready
                except Exception as exc:
                    ready[str(port)] = dict(error=repr(exc))
        dump(directory/'worker_ready.json', ready)
        if not clients:
            raise RuntimeError('No worker initialized; stopping rather than producing an empty matrix')
        summary['idle_start_monotonic'] = time.monotonic()
        # Identical fixed pre-trial settling window. No artificial claim that
        # every trial reaches the same starting temperature.
        deadline = time.monotonic()+spec['idle_s']
        while time.monotonic() < deadline:
            if sampler.errors:
                raise RuntimeError('PoE collection failed during pre-trial idle')
            time.sleep(min(.25, max(0., deadline-time.monotonic())))
        for port in spec.get('ports', COHORTS.get(spec['cohort'], [])):
            try:
                initial[str(port)] = clients[port].probe('pretrial')
            except Exception as exc:
                initial[str(port)] = dict(telemetry={'cpu_c': None}, error=repr(exc))
        dump(directory/'initial_status.json', initial)
        nodes = node_builder(spec['cohort'], datasets['service'], datasets['transport'],
                            datasets['power'], initial)
        policy = PhysicalPolicy(nodes, spec['method'], spec['budget_factor'],
            v_joules=spec['v_joules'], energy_margin=spec['energy_margin'],
            constraint_mode=spec.get('constraint_mode', 'legacy'))
        dump(directory/'model_parameters.json', dict(nodes=[asdict(n) for n in nodes],
            budget_j_per_minute=policy.budget.tolist(),
            power_scope='Incremental task charge above separately measured/modelled idle baseline',
            full_fleet_energy_claim='No measured total for externally powered boards',
            thermal_guard_is_algorithm_independent=True, constraint_mode=policy.constraint_mode,
            energy_guard=policy.guard, request_guard=policy.request_guard))
        # Compile/cache shared primitives before the trial clock starts.
        warm = PhysicalPolicy(node_builder(spec['cohort'], datasets['service'],
            datasets['transport'], datasets['power'], initial), spec['method'],
            spec['budget_factor'], v_joules=spec['v_joules'], energy_margin=spec['energy_margin'],
            constraint_mode=spec.get('constraint_mode', 'legacy'))
        warm.decide(jobs[0], 0.)
        for port, client in clients.items():
            pipelines[port] = BoardPipeline(client=client)
        completed = queue.Queue()
        outstanding = {}
        origin = time.monotonic()
        summary['arrival_start_monotonic'] = origin
        summary['arrival_end_monotonic'] = origin+spec['duration_s']
        summary['measurement_end_monotonic'] = origin+spec['duration_s']+spec['drain_s']
        with (directory/'events.jsonl').open('x', encoding='utf-8') as log:
            def receive_results():
                while True:
                    try:
                        identity, index, future = completed.get_nowait()
                    except queue.Empty:
                        return
                    job, reference = outstanding.pop(identity)
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = dict(id=identity, status='transport_error', error=repr(exc),
                                      after={'cpu_c': None}, predictions=[],
                                      controller_received=time.monotonic())
                    actual = np.asarray(result.get('predictions', []), dtype=float)
                    if len(actual) > len(reference):
                        raise ValueError('Worker returned too many predictions')
                    parity = bool(np.all(np.isfinite(actual)) and (not len(actual) or
                        np.max(np.abs(actual-reference[:len(actual)])/np.maximum(
                            np.abs(reference[:len(actual)]), 1.)) < 1e-4))
                    result['parity_ok'] = parity
                    result['release_to_result_s'] = result['controller_received']-origin-job['release_s']
                    result['observed_by_policy_s'] = time.monotonic()-origin
                    result['port'] = nodes[index].port
                    result['batch_atomic_success'] = (result['status']=='ok' and parity
                                                       and len(actual)==job['batch'])
                    policy.completed(index, result, result['observed_by_policy_s'])
                    if result['batch_atomic_success']:
                        summary['successful_batches'] += 1
                        summary['successful_inferences'] += job['batch']
                    elif result['status'] == 'temperature_guard':
                        summary['guarded_batches'] += 1
                    else:
                        summary['failed_batches'] += 1
                    summary['returned_individual_predictions'] += len(actual)
                    summary['parity_failures'] += int(not parity)
                    record(log, dict(event='result', job=job, result=result))
                    if not parity:
                        raise RuntimeError('Numerical parity failed; preserving raw evidence and stopping')

            cursor, next_state = 0, 0.
            while time.monotonic() < summary['measurement_end_monotonic']:
                if sampler.errors:
                    raise RuntimeError('PoE collection stopped; incomplete trial is retained')
                receive_results()
                now = time.monotonic()-origin
                if now >= next_state:
                    policy.advance(now)
                    record(log, dict(event='state', observed_s=now, debt_j=policy.debt.tolist(),
                        credit_j=policy.credit.tolist(), tenant_debt_j=policy.tenant_debt.tolist(),
                        tenant_credit_j=policy.tenant_credit.tolist(),
                        pending=[len(n.pending) for n in nodes], healthy=[n.healthy for n in nodes],
                        temperature_c=[n.temperature_c for n in nodes]))
                    next_state = now+1.
                if cursor < len(jobs) and now >= jobs[cursor]['release_s']:
                    job = jobs[cursor]
                    cursor += 1
                    if now >= spec['duration_s']:
                        summary['rejected_batches'] += 1
                        record(log, dict(event='rejected', job=job, observed_s=now,
                                         reason='controller_arrival_horizon_overrun'))
                        continue
                    began = time.perf_counter()
                    decision = policy.decide(job, now)
                    decision_s = time.perf_counter()-began
                    index = decision['selected']
                    event = dict(event='decision', job=job, observed_s=now,
                                 decision_s=decision_s, decision=decision)
                    if index is None:
                        summary['rejected_batches'] += 1
                    else:
                        features, expected = select_service_batch(examples, job)
                        try:
                            future = pipelines[nodes[index].port].submit(job['id'], features,
                                                                        job['payload_bytes'],
                                                                        service_id=job.get('service_id', 'mlp'))
                        except Exception as exc:
                            nodes[index].healthy = False
                            event['submission_error'] = repr(exc)
                            summary['rejected_batches'] += 1
                        else:
                            policy.commit(job, decision)
                            outstanding[job['id']] = (job, expected)
                            future.add_done_callback(lambda f, identity=job['id'], idx=index:
                                                     completed.put((identity, idx, f)))
                            summary['dispatched_batches'] += 1
                    event['dispatch_path_s'] = time.perf_counter()-began
                    record(log, event)
                else:
                    upcoming = jobs[cursor]['release_s']-now if cursor < len(jobs) else .005
                    time.sleep(max(0., min(.005, upcoming)))
            receive_results()
            # No silently omitted offered jobs or right-censored success.
            for job in jobs[cursor:]:
                summary['rejected_batches'] += 1
                record(log, dict(event='rejected', job=job, observed_s=time.monotonic()-origin,
                                 reason='controller_arrival_horizon_overrun'))
            summary['outstanding_at_fixed_drain'] = sorted(outstanding)
            for identity, (job, _) in outstanding.items():
                record(log, dict(event='censored_failure', job=job,
                                 reason='not returned within fixed arrival-plus-drain window'))
            policy.advance(time.monotonic()-origin)
            summary['final_debt_j'] = policy.debt.tolist()
            summary['final_credit_j'] = policy.credit.tolist()
            summary['ports'] = [n.port for n in nodes]
            summary['end_healthy'] = [n.healthy for n in nodes]
            summary['complete'] = True
    except Exception as exc:
        summary['error'] = repr(exc)
        raise
    finally:
        for port, client in clients.items():
            summary['cleanup'][str(port)] = (pipelines[port].close() if port in pipelines else client.close())
        summary['ended_utc'] = utc()
        dump(directory/'summary.json', summary)
    if any(c.get('errors') or not c.get('stop_acknowledged') or not c.get('directory_removed')
           for c in summary['cleanup'].values()):
        raise RuntimeError('Worker cleanup not confirmed; stopping before a contaminated next trial')
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--transport', type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=10)
    parser.add_argument('--model', type=Path, default=Path('results/revision/physical/portable_model_20260905'))
    parser.add_argument('--service', type=Path, default=Path('results/revision/physical/inference_calibration_v2_20260905/service_models.json'))
    parser.add_argument('--power', type=Path, default=Path('results/revision/physical/power_thermal_20260905/analysis.json'))
    parser.add_argument('--aggregate', type=Path, default=Path('results/revision/workload_aggregate_16x4.npz'))
    parser.add_argument('--forecast', type=Path, default=Path('results/revision/forecast_prequential_20260905/forecast_prequential.npz'))
    args = parser.parse_args()
    if str(args.out).startswith('\\\\'):
        raise ValueError('Collect high-frequency trial logs on local disk, archive to NAS afterward')
    args.out.mkdir(parents=True, exist_ok=False)
    trials = design(args.repetitions)
    hashes = source_hashes()
    inputs = {name: getattr(args, name) for name in
              ['inventory', 'service', 'power', 'transport', 'aggregate', 'forecast']}
    inputs['model'] = args.model/'model.json'
    inputs['features'] = args.model/'validation_inputs.npz'
    manifest = dict(started_utc=utc(), trials=trials, planned_trials=len(trials),
        source_sha256=hashes,
        input_sha256={name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in inputs.items()},
        design='Ten randomized matched blocks; identical offered jobs across policies within cohort/scenario/block',
        primary_metrics=['offered on-time request success', 'request-weighted latency and P95',
                         'measured PSE supplied energy for five-port cohort only', 'peak observed CPU temperature'],
        energy_boundary='PoE five-port sum, common 75-s arrival-plus-drain window; idle measured separately',
        no_adaptation='No policy tuning or workload selection based on interim outcomes',
        isolation='No simultaneous calibration, stress simulation, firmware change or external traffic injection',
        power_session_protocol='Fresh authenticated read-only session before every trial; no reconnect/retry inside a trial or interpolation across missing samples',
        workload='Trace-shaped new CPU-MLP requests, not the lost application portfolio',
        unmetered_assumptions='Fleet transfers and RDK/Nano coefficients are explicit in each model_parameters.json',
        temperature_guard_c=65., statistics_unit='Matched trial block, not individual requests or 1-Hz samples')
    dump(args.out/'manifest.json', manifest)
    datasets = {name: json.loads(inputs[name].read_text(encoding='utf-8'))
                for name in ['service', 'power', 'transport']}
    datasets.update(aggregate=dict(np.load(args.aggregate)), forecast=dict(np.load(args.forecast)))
    if 'One board at a time' not in datasets['transport'].get('controller_condition', ''):
        raise ValueError('Formal trials require the isolated controller calibration')
    inventory = json.loads(args.inventory.read_text(encoding='utf-8'))['ports']
    model = json.loads(inputs['model'].read_text(encoding='utf-8'))
    examples = dict(np.load(inputs['features']))
    key = os.environ['EDGE_SSH_KEY']
    password = getpass.getpass('Switch password (hidden): ')
    sampler, completed = None, []
    try:
        sampler = PowerSampler(args.out, password)
        password = None
        for spec in trials:
            check_sources(hashes)
            if spec['ordinal']:
                sampler.renew_between_trials()
            rows = [r for r in inventory if r['port'] in COHORTS[spec['cohort']]]
            result = run_trial(spec, args.out/('trial_%03d' % spec['ordinal']),
                               rows, key, model, examples, datasets, sampler)
            completed.append(dict(ordinal=spec['ordinal'], complete=result['complete']))
            print(json.dumps(dict(event='trial_complete', ordinal=spec['ordinal'],
                total=len(trials), block=spec['block'], cohort=spec['cohort'], scenario=spec['scenario'],
                method=spec['method'], successful=result['successful_batches'], rejected=result['rejected_batches'],
                guarded=result['guarded_batches'], error=result['error'])), flush=True)
    finally:
        password = None
        dump(args.out/'campaign_status.json', dict(completed=completed, ended_utc=utc(),
            complete=len(completed)==len(trials), power_sampler=sampler.close() if sampler else None))


if __name__ == '__main__':
    main()
