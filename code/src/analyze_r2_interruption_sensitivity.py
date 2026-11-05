"""Descriptive service sensitivity retaining complete conditions from retries.

The primary estimate remains the ten full registered blocks. This diagnostic
adds every fully completed condition (all its policies) from interrupted
attempts, averages repeated attempts within their original block, and keeps
ten blocks as the statistical units. Power is excluded because acquisition
failure is exactly why these attempts could not enter the primary matrix.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from src.analyze_r2_physical import PRIMARY, interval, sha
from src.summarize_physical_trials import weighted_quantile


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def service_record(directory, spec):
    summary = read(directory/'summary.json')
    protocol = read(directory/'protocol.json')
    params = read(directory/'model_parameters.json')
    jobs = read(directory/'offered_jobs.json')
    if not summary['complete'] or summary['error'] or summary['parity_failures']:
        raise ValueError('Only fully completed trials support the diagnostic')
    if not params['energy_guard'] or not params['request_guard'] or params['constraint_mode'] != 'matched':
        raise ValueError('Changed resource constraints')
    jobhash = hashlib.sha256(json.dumps(jobs, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if jobhash != protocol['offered_sha256']:
        raise ValueError('Offered job hash mismatch')
    results = {}
    rejected = set()
    assigned = set()
    with (directory/'events.jsonl').open(encoding='utf-8') as incoming:
        for line in incoming:
            event = json.loads(line)
            if event['event'] == 'decision':
                identity = event['job']['id']
                if identity in rejected or identity in assigned:
                    raise ValueError('Duplicate dispatch decision')
                if event['decision']['selected'] is None or 'submission_error' in event:
                    rejected.add(identity)
                else:
                    assigned.add(identity)
            elif event['event'] == 'rejected':
                identity = event['job']['id']
                if identity in rejected or identity in assigned:
                    raise ValueError('Duplicate rejection')
                rejected.add(identity)
            elif event['event'] == 'result':
                identity = event['job']['id']
                if identity in results:
                    raise ValueError('Duplicate result')
                results[identity] = event['result']
    offered = {j['id'] for j in jobs}
    if len(offered) != 2880 or assigned | rejected != offered or not set(results) <= assigned:
        raise ValueError('Offered-work conservation failure')
    if assigned - set(results) != set(summary['outstanding_at_fixed_drain']):
        raise ValueError('Censoring record mismatch')
    total = sum(j['batch'] for j in jobs)
    success = complete = 0
    latencies, weights = [], []
    for job in jobs:
        result = results.get(job['id'])
        if result and result['batch_atomic_success']:
            if not result['parity_ok'] or len(result['predictions']) != job['batch']:
                raise ValueError('Prediction parity failure')
            delay = result['release_to_result_s']
            complete += job['batch']
            success += job['batch'] * (delay <= job['deadline_s'])
            latencies.append(delay)
            weights.append(job['batch'])
    if total != protocol['offered_inferences'] or complete != summary['successful_inferences']:
        raise ValueError('Inference totals disagree')
    return dict(offered_on_time_success=success/total,
        rejection_fraction=sum(j['batch'] for j in jobs if j['id'] in rejected)/total,
        mean_latency_ms=float(np.average(latencies, weights=weights)*1000) if complete else float('nan'),
        p95_latency_ms=weighted_quantile(latencies, weights, .95)*1000 if complete else float('nan'),
        offered_sha256=jobhash, budget=params['budget_j_per_minute'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--formal', type=Path, required=True)
    parser.add_argument('--interruptions', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    manifest = read(args.formal/'manifest.json')
    final = read(args.formal/'campaign_status.json')
    if not final['complete'] or final['completed'] != list(range(140)):
        raise ValueError('Complete the full primary matrix first')
    primary = pd.read_csv(args.formal/'analysis/trial_records.csv')
    if len(primary) != 140:
        raise ValueError('Primary analysis must be complete')
    args.out.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    for spec in manifest['trials']:
        grouped[spec['block'], spec['scenario'], spec['workload']].append(spec)
    records = []
    decisions = []
    evidence = {}
    for _, row in primary.iterrows():
        records.append(dict(attempt='primary', block=int(row.block), scenario=row.scenario,
            workload=row.workload, method=row.method, **{m:row[m] for m in PRIMARY}))
    for parent in sorted(args.interruptions.iterdir()):
        if not parent.is_dir():
            continue
        for key, specs in grouped.items():
            paths = [parent/('trial_%03d'%s['ordinal']) for s in specs]
            present = [p for p in paths if (p/'summary.json').is_file()]
            if not present:
                continue
            complete = len(present) == len(paths) and all(read(p/'summary.json')['complete'] for p in present)
            decisions.append(dict(attempt=parent.name, block=key[0], scenario=key[1], workload=key[2],
                included=complete, reason='all policies complete' if complete else 'condition incomplete as a whole'))
            if not complete:
                continue
            for spec, path in zip(specs, paths):
                row = service_record(path, spec)
                reference = args.formal/path.name
                expected = read(reference/'protocol.json')
                parameters = read(reference/'model_parameters.json')
                if row['offered_sha256'] != expected['offered_sha256'] or row['budget'] != parameters['budget_j_per_minute']:
                    raise ValueError('Repeated condition changed jobs or budgets')
                records.append(dict(attempt=parent.name, block=key[0], scenario=key[1], workload=key[2],
                    method=spec['method'], **{m:row[m] for m in PRIMARY}))
                for name in ['summary.json','protocol.json','model_parameters.json','offered_jobs.json','events.jsonl']:
                    evidence[(parent.name+'/'+path.name+'/'+name)] = sha(path/name)
    frame = pd.DataFrame(records)
    frame.to_csv(args.out/'all_complete_condition_records.csv', index=False)
    # Missing outcomes propagate. Do not silently average only finite repeats.
    collapsed = frame.groupby(['block','scenario','workload','method'])[PRIMARY].agg(lambda x:np.mean(x.to_numpy(float))).reset_index()
    collapsed.to_csv(args.out/'block_means.csv', index=False)
    contrasts = []
    for (scenario, workload, method), challenger in collapsed.groupby(['scenario','workload','method']):
        if method == 'scd':
            continue
        reference = collapsed[(collapsed.scenario==scenario)&(collapsed.workload==workload)&(collapsed.method=='scd')].set_index('block').sort_index()
        challenger = challenger.set_index('block').loc[reference.index]
        original = primary[(primary.scenario==scenario)&(primary.workload==workload)]
        a = original[original.method==method].set_index('block').sort_index()
        b = original[original.method=='scd'].set_index('block').loc[a.index]
        for metric in PRIMARY:
            mean, half, n = interval(challenger[metric]-reference[metric])
            initial, initial_half, _ = interval(a[metric]-b[metric])
            contrasts.append(dict(scenario=scenario, workload=workload, challenger=method, metric=metric,
                primary_mean=initial, primary_ci95=initial_half, sensitivity_mean=mean, sensitivity_ci95=half,
                estimate_shift=mean-initial, available_blocks=n, contrast='challenger minus SCD'))
    pd.DataFrame(contrasts).to_csv(args.out/'paired_effects.csv', index=False)
    shutil.copy2(Path(__file__), args.out/'analysis_source.py')
    report = dict(complete=True, primary_trials=140, additional_complete_trials=len(frame)-140,
        complete_interrupted_conditions=sum(d['included'] for d in decisions), decisions=decisions,
        source_sha256=evidence, analysis_source_sha256=sha(Path(__file__)),
        scope='Descriptive sensitivity, not a replacement primary analysis or an additional significance family.',
        aggregation='Every complete condition from every interrupted attempt is added. Repeats are averaged within each original block before ten-block paired intervals.',
        power='Excluded because complete service outcomes cannot repair unobserved power.')
    (args.out/'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['decisions','source_sha256']}, indent=2))


if __name__=='__main__':
    main()
