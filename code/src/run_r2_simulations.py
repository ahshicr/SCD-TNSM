"""R2 full-trace factorial. Freeze this protocol before inspecting new results."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import pandas as pd
from src.run_strict_campaign import add_task, _run_trace_group, execution_source_hashes, runtime_environment
from src.strict_simulate import StrictScenario
from src.strict_config import load_deployment

METHODS = ['scd', 'dpp_matched', 'compute_first', 'scd_no_debt',
           'scd_no_debt_feedback', 'scd_no_forecast', 'scd_static_weights']
CONDITIONS = [
    dict(name='r2_nominal'),
    dict(name='r2_energy_03', energy_budget_factor=.3),
    dict(name='r2_energy_01', energy_budget_factor=.1),
    dict(name='r2_network_100', inter_switch_rate_mbps=100., background_multiplier=12., payload_multiplier=4.),
]

def formal_tasks(root):
    tasks = []
    for seed in range(20260914, 20260919):
        trace = root / f'binomial_p0004_seed{seed}'
        base = dict(seed=seed, start_day=10, end_day_exclusive=14)
        for condition in CONDITIONS:
            for method in METHODS:
                add_task(tasks, trace, method, StrictScenario(**condition, **base), 'r2_components')
            for method in ['scd_shared_energy', 'scd_elastic_energy']:
                add_task(tasks, trace, method, StrictScenario(**condition, **base), 'r2_energy_allocation')
            add_task(tasks, trace, 'scd', StrictScenario(**{**condition, 'name': condition['name']+'_no_energy_guard'},
                     **base, energy_guard=False), 'r2_energy_guard')
    return tasks

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--trace-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=5)
    p.add_argument('--deployment', type=Path, default=Path('config/strict_deployment_20260914.json'))
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    tasks = formal_tasks(a.trace_root)
    deployment = load_deployment(a.deployment)
    hashes = execution_source_hashes(Path(__file__))
    protocol = dict(schema='tnsm.r2.full-trace.v1', tasks=tasks, task_count=len(tasks),
        execution_source_sha256=hashes, deployment_sha256=deployment.source_sha256,
        forecast_sha256=hashlib.sha256(deployment.service_forecast_path.read_bytes()).hexdigest(),
        runtime_environment=runtime_environment(),
        evaluation='Full 14-day trace, state advanced from day 0, outcomes on days 10 to 13 inclusive, no truncation',
        statistics='Five paired seeds, Student t 95% CI. Holm correction within all prespecified contrasts. All outcomes reported.',
        primary_metrics=['success_ratio', 'task_energy_mj', 'supplied_energy_mj', 'p95_latency_ms', 'mean_peak_aoi_ms', 'jain_tenant_success', 'minimum_tenant_success_ratio'],
        contrasts='SCD versus each component variant in each condition. Elastic energy versus fixed SCD and shared energy in each condition. No energy guard versus SCD.',
        elastic_semantics='Only device eligibility is hard. Tenant debt pricing and separate request tokens remain. Negative tenant credit is borrowed capacity and repaid by the unchanged tenant refill rate. No strict tenant burst envelope is claimed.',
        selection_policy='No condition, method, seed, or metric dropped on the basis of observed outcome')
    path = a.out/'protocol.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != protocol:
        raise ValueError('Protocol changed. Use a new output directory.')
    if not path.exists():
        path.write_text(json.dumps(protocol, indent=2), encoding='utf-8')
        (a.out/'started.json').write_text(json.dumps({'utc': datetime.now(timezone.utc).isoformat()}), encoding='utf-8')
        snap = a.out/'source_snapshot'
        snap.mkdir()
        for name in hashes:
            shutil.copy2(Path(__file__).with_name(name), snap/name)
    groups = {}
    for task in tasks:
        groups.setdefault(task['trace'], []).append(task)
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures = {pool.submit(_run_trace_group, group, str(a.deployment), str(a.out)): key for key, group in groups.items()}
        for future in as_completed(futures):
            rows.extend(future.result())
            pd.json_normalize(rows, sep='.').sort_values('key').to_csv(a.out/'campaign_results.csv', index=False)
            print(json.dumps({'finished': len(rows), 'planned': len(tasks)}), flush=True)
    if len(rows) != len(tasks):
        raise AssertionError('Incomplete factorial')
    (a.out/'completion.json').write_text(json.dumps({'complete': True, 'runs': len(rows), 'utc': datetime.now(timezone.utc).isoformat()}), encoding='utf-8')

if __name__ == '__main__':
    main()
