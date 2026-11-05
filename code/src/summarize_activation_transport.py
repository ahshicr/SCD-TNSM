"""Separate service, controller overhead and same-clock measured overlap."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import t


def stats(values):
    a = np.asarray(values, dtype=float)
    return dict(n=len(a), mean=float(a.mean()), median=float(np.median(a)),
                p95=float(np.quantile(a, .95)), minimum=float(a.min()), maximum=float(a.max()),
                ci95_half=float(t.ppf(.975, len(a)-1)*a.std(ddof=1)/np.sqrt(len(a))) if len(a)>1 else None)


def overlap_with_compute(left, right, jobs):
    return sum(max(0., min(right, j['compute_end'])-max(left, j['compute_start']))
               for j in jobs if 'compute_start' in j)


def lines(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('directory', type=Path)
    args = p.parse_args()
    directory = args.directory
    summary = json.loads((directory/'summary.json').read_text())
    protocol = json.loads((directory/'protocol.json').read_text())
    activation, transport, raw = [], [], []
    for board in summary['activation']:
        jobs = lines(directory/('port_%02d_activation.jsonl' % board['port']))
        completed = [j for j in jobs if j['status'] == 'ok']
        if len(completed) != 20 or any(j['cleanup']['errors'] or not j['cleanup']['stop_acknowledged']
                                     or not j['cleanup']['directory_removed'] for j in completed):
            raise ValueError('Incomplete activation or cleanup')
        activation.append(dict(port=board['port'],
            worker_activation_s=stats([j['worker_ready']['controller_startup_s'] for j in completed]),
            provision_plus_activation_s=stats([j['provision_and_activation_s'] for j in completed]),
            model_shape_validation_s=stats([j['worker_ready']['validation_s'] for j in completed])))
    for board in summary['transport']:
        path = directory/('port_%02d_transport.jsonl' % board['port'])
        if not path.exists():
            continue
        jobs = lines(path)
        for j in jobs:
            row = dict(port=board['port'], condition=j['condition'], depth=j['pipeline_depth'],
                payload_bytes=j['target_payload_bytes'], repetition=j['repetition'],
                status=j['status'], parity_ok=j['parity_ok'],
                controller_latency_s=j['controller_latency_s'])
            if 'compute_start' in j:
                row.update(service_s=j['compute_end']-j['compute_start'],
                    remote_payload_s=j['payload_received']-j['receive_start'],
                    remote_queue_s=max(0., j['compute_start']-j['payload_received']),
                    payload_compute_overlap_s=overlap_with_compute(j['receive_start'], j['payload_received'], jobs),
                    controller_residual_s=j['controller_latency_s']-(j['compute_end']-j['compute_start']))
            raw.append(row)
        own = pd.DataFrame([r for r in raw if r['port'] == board['port']])
        conditions = []
        for (payload, depth), group in own.groupby(['payload_bytes', 'depth']):
            selected = [j for j in jobs if j['target_payload_bytes']==payload and j['pipeline_depth']==depth]
            successful = group[(group.status == 'ok')&group.parity_ok]
            if len(successful) != len(group):
                conditions.append(dict(payload_bytes=int(payload), depth=int(depth), complete=False))
                continue
            span = max(j['compute_end'] for j in selected)-min(j['receive_start'] for j in selected)
            conditions.append(dict(payload_bytes=int(payload), depth=int(depth), complete=len(group)==20,
                service_s=stats(group.service_s), controller_latency_s=stats(group.controller_latency_s),
                remote_queue_s=stats(group.remote_queue_s), remote_payload_s=stats(group.remote_payload_s),
                controller_residual_s=stats(group.controller_residual_s),
                overlap_s=float(group.payload_compute_overlap_s.sum()),
                overlapped_payloads=int((group.payload_compute_overlap_s > 0).sum()),
                receive_compute_span_s=span, inferences_per_span_s=16*len(group)/span))
        fit_rows = own[(own.depth==1)&(own.status=='ok')&own.parity_ok]
        medians = fit_rows.groupby('payload_bytes').controller_residual_s.median()
        network_fit = None
        if len(medians) == 4:
            # Bytes/MB avoids numerical imbalance of the nonnegative regressors.
            x = np.column_stack([np.ones(len(medians)), medians.index.to_numpy()/1e6])
            coef, _ = nnls(x, medians.to_numpy())
            network_fit = dict(fixed_overhead_s=float(coef[0]), seconds_per_decimal_mb=float(coef[1]),
                effective_mbps=float(8/coef[1]) if coef[1] > 0 else None,
                median_point_rmse_s=float(np.sqrt(np.mean((x@coef-medians.to_numpy())**2))),
                scope='Nonnegative fit to depth-one controller latency minus remote compute, not pure link capacity')
        cleanup = board.get('cleanup', {})
        if cleanup.get('errors') or not cleanup.get('stop_acknowledged') or not cleanup.get('pipeline_threads_stopped'):
            raise ValueError('Transport cleanup not confirmed')
        transport.append(dict(port=board['port'], complete=board['complete'], conditions=conditions,
                              controller_transport_fit=network_fit))
    pd.DataFrame(raw).to_csv(directory/'transport_timing_records.csv', index=False)
    result = dict(activation=activation, transport=transport,
        observation_scope='Within-session repeated measurements; concurrent declared board cohort',
        overlap_clock='Receive and compute endpoints are from the same remote monotonic clock',
        controller_condition=protocol.get('controller_condition',
            'Collected concurrently with a six-worker local simulation campaign; repeat idle-controller transport before final policy calibration'),
        inference_scope='Actual trained CPU MLP, batch 16; not GPU/NPU or cryptographic proxy',
        source_summary_sha256=hashlib.sha256((directory/'summary.json').read_bytes()).hexdigest())
    (directory/'activation_transport_analysis.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    for b in activation:
        print(json.dumps(dict(port=b['port'], activation_ms=b['worker_activation_s']['mean']*1000)))
    for b in transport:
        print(json.dumps(dict(port=b['port'], complete=b['complete'],
            overlap_depth1=sum(c.get('overlap_s', 0) for c in b['conditions'] if c['depth']==1),
            overlap_depth4=sum(c.get('overlap_s', 0) for c in b['conditions'] if c['depth']==4),
            network_fit=b['controller_transport_fit'])))


if __name__ == '__main__':
    main()
