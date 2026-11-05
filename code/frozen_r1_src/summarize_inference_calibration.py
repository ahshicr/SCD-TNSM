"""Source-backed per-board service calibration, preserving guarded/incomplete nodes."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t


def interval(values):
    x = np.asarray(values, dtype=float)
    return dict(n=len(x), mean=float(x.mean()), median=float(np.median(x)),
                sd=float(x.std(ddof=1)) if len(x) > 1 else None,
                ci95_half=float(t.ppf(.975, len(x)-1)*x.std(ddof=1)/np.sqrt(len(x))) if len(x)>1 else None,
                p95=float(np.quantile(x, .95)), minimum=float(x.min()), maximum=float(x.max()))


def summarize(directory):
    completed = json.loads((directory/'summary.json').read_text(encoding='utf-8'))
    report = dict(scope='CPU reference MLP calibration, not GPU/NPU throughput; repetitions within one session',
                  confidence='Student-t intervals across repetitions; not independent trial-level policy CIs',
                  source_summary_sha256=hashlib.sha256((directory/'summary.json').read_bytes()).hexdigest(),
                  boards=[], batch_statistics=[])
    records = []
    for board in completed['boards']:
        port = board['port']
        source = directory/('port_%02d.jsonl' % port)
        raw = [json.loads(s) for s in source.read_text(encoding='utf-8').splitlines()] if source.exists() else []
        ready = board.get('worker_ready', {})
        entry = dict(port=port, model=ready.get('board'), arch=ready.get('arch'),
                     complete=board['complete'], successful_jobs=board['succeeded'],
                     guarded_jobs=board['guarded'], failed_jobs=board['failed'],
                     startup_s=ready.get('controller_startup_s'),
                     source_sha256=ready.get('source_sha256'), cleanup=board.get('cleanup'),
                     initial_cpu_c=ready.get('telemetry', {}).get('cpu_c'), batch_models={})
        for row in raw:
            if row['status'] != 'ok' or not row.get('parity_ok'):
                continue
            service = row['compute_end']-row['compute_start']
            # CPU/inference interval still includes OS scheduling and sampler
            # contention. Subtracting measured checks is only a diagnostic.
            record = dict(port=port, batch=row['batch'], repetition=row['repetition'],
                          service_s=service, safety_check_s=row['safety_check_s'],
                          check_fraction=row['safety_check_s']/max(service, 1e-12),
                          inference_excluding_checks_s=service-row['safety_check_s'],
                          controller_latency_s=row['controller_latency_s'],
                          outside_compute_s=row['controller_latency_s']-service,
                          cpu_before=row['before']['cpu_c'], cpu_after=row['after']['cpu_c'],
                          max_relative_error=row['max_relative_error'])
            records.append(record)
        own = [r for r in records if r['port'] == port]
        for batch in (1, 4, 16, 64):
            rows = [r for r in own if r['batch'] == batch]
            if not rows:
                continue
            stats = dict(port=port, batch=batch,
                         **{metric: interval([r[metric] for r in rows])
                            for metric in ('service_s', 'safety_check_s', 'check_fraction',
                                           'inference_excluding_checks_s', 'controller_latency_s',
                                           'outside_compute_s', 'cpu_before', 'cpu_after')})
            report['batch_statistics'].append(stats)
            entry['batch_models'][str(batch)] = dict(service_mean_s=stats['service_s']['mean'],
                service_p95_s=stats['service_s']['p95'], observations=len(rows),
                max_relative_error=max(r['max_relative_error'] for r in rows),
                mean_safety_fraction=stats['check_fraction']['mean'])
        report['boards'].append(entry)
    report['successful_jobs'] = len(records)
    report['complete_boards'] = sum(r['complete'] for r in report['boards'])
    report['successful_individual_inferences'] = sum(r['batch'] for r in records)
    pd.DataFrame(records).to_csv(directory/'timing_records.csv', index=False)
    (directory/'service_models.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('successful_jobs', 'complete_boards', 'successful_individual_inferences')}))
    for row in report['boards']:
        print(json.dumps(dict(port=row['port'], complete=row['complete'],
             batch16=row['batch_models'].get('16'), batch64=row['batch_models'].get('64'))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    summarize(parser.parse_args().directory)
