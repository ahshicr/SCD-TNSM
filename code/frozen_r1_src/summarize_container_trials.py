"""Complete-block startup statistics; failed network acquisitions stay explicit."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

from src.container_trial_design import BLOCKS, PORTS, PULLS, RESIDENT_CALLS, design


def interval(values):
    data = np.asarray(values, dtype=float)
    if len(data) != BLOCKS or not np.isfinite(data).all():
        raise ValueError('Exactly twenty complete finite block observations are required')
    return dict(mean=float(data.mean()), halfwidth95=float(t.ppf(.975, BLOCKS-1)*
        data.std(ddof=1)/np.sqrt(BLOCKS)), n=BLOCKS)


def audit_activation(observation, spec):
    if observation.get('spec') != spec or not observation.get('complete') or observation.get('error'):
        raise ValueError('Activation identity or completion mismatch')
    clean = observation.get('cleanup', {})
    if not clean.get('stop_acknowledged') or clean.get('errors') or clean.get('exit_status') != 0:
        raise ValueError('Activation cleanup incomplete')
    requests = observation['requests']
    if len(requests) != 1+RESIDENT_CALLS:
        raise ValueError('Missing first/resident inference results')
    indices = [spec['first_input']]+spec['resident_inputs']
    for identity, (request, index) in enumerate(zip(requests, indices)):
        if (request['id'] != identity or request['input_index'] != index or request['status'] != 'ok'
                or request['computed'] != 1):
            raise ValueError('Request identity/service mismatch')
        parity = request.get('max_relative_error')
        if parity is None or not np.isfinite(parity) or not 0 <= parity < 1e-4:
            raise ValueError('Invalid numerical parity')
        for side in ['before', 'after']:
            temp = request[side]['cpu_c']
            if temp is None or not np.isfinite(temp) or not -20 <= temp < 65:
                raise ValueError('Missing/unsafe measured CPU temperature')
        if not (request['controller_send_start'] <= request['controller_send_end'] <=
                request['controller_received']):
            raise ValueError('Invalid controller timing order')
        if not (request['compute_start'] <= request['compute_end']):
            raise ValueError('Invalid remote service timing order')
    metrics = dict(ready_ms=1000*observation['launch_to_ready_s'],
        first_prediction_ms=1000*observation['launch_to_first_prediction_s'],
        resident_response_ms=1000*np.mean([r['controller_latency_s'] for r in requests[1:]]),
        resident_compute_ms=1000*np.mean([r['compute_end']-r['compute_start'] for r in requests[1:]]),
        model_send_ms=1000*observation['model_send_s'])
    if not all(np.isfinite(v) and v >= 0 for v in metrics.values()):
        raise ValueError('Invalid activation metric')
    if metrics['first_prediction_ms'] < metrics['ready_ms']:
        raise ValueError('First result must follow deployment readiness')
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('campaign', type=Path)
    args = parser.parse_args()
    root = args.campaign
    status = json.loads((root/'campaign_status.json').read_text(encoding='utf-8'))
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    if not status.get('complete') or manifest['trials'] != design():
        raise ValueError('Full fixed container protocol is incomplete or changed')
    if sorted(b['port'] for b in status['boards']) != sorted(PORTS):
        raise ValueError('Missing or duplicate physical platform')
    boards, raw_checksums = [], {}
    for port in PORTS:
        directory = root/('port_%02d' % port)
        board = next(b for b in status['boards'] if b['port'] == port)
        if not board.get('complete') or board.get('error') or board.get('cleanup_error') or board.get('evidence_error'):
            raise ValueError('Board protocol/cleanup incomplete')
        if board['activations'] != 2*BLOCKS or len(board['cleanup']) != PULLS:
            raise ValueError('Incomplete runtime acquisition/activation repetitions')
        for clean in board['cleanup']:
            if (not clean.get('daemon_stopped') or clean.get('remaining_mounts') or
                    clean.get('remaining_store_processes') or not clean.get('forwarding_unchanged')):
                raise ValueError('Private runtime shutdown/state verification failed')
        evidence = directory/'runtime_evidence'
        collection = json.loads((evidence/'collection.json').read_text(encoding='utf-8'))
        if collection.get('failures'):
            raise ValueError('Runtime evidence copy failed')
        for entry in collection['copied']:
            if Path(entry['local']).name != entry['local']:
                raise ValueError('Invalid evidence filename')
            path = evidence/entry['local']
            if (path.stat().st_size != entry['bytes'] or
                    hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']):
                raise ValueError('Runtime evidence differs from its collected hash')
        by_method = {m: [] for m in ['host_process', 'cached_container']}
        versions = {m: set() for m in by_method}
        for spec in (s for s in manifest['trials'] if s['port'] == port):
            path = directory/('activation_b%02d_%s.json' % (spec['block'], spec['method']))
            raw_checksums[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
            obs = json.loads(path.read_text(encoding='utf-8'))
            by_method[spec['method']].append(audit_activation(obs, spec))
            versions[spec['method']].add(obs['worker_ready']['python'])
            for name in ['board_inference_worker.py', 'portable_inference.py']:
                if obs['worker_ready']['source_sha256'][name] != manifest['source_sha256'][name]:
                    raise ValueError('Actual worker source identity mismatch')
        metrics = {}
        for name in by_method['host_process'][0]:
            host = [r[name] for r in by_method['host_process']]
            container = [r[name] for r in by_method['cached_container']]
            metrics[name] = dict(host_process=interval(host), cached_container=interval(container),
                                paired_container_minus_host=interval(np.asarray(container)-host))
        acquisitions = []
        for i in range(PULLS):
            path = directory/'runtime_evidence'/('pull_%02d_acquisition.json' % i)
            if not path.exists():
                raise ValueError('Missing raw acquisition record, including failed attempts')
            raw_checksums[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
            obs = json.loads(path.read_text(encoding='utf-8'))
            result = obs['download']
            acquisitions.append(dict(repetition=i, success=result['returncode'] == 0,
                elapsed_s=result['elapsed_s'], timeout_s=result.get('timeout_s'),
                reference=obs['reference'], stderr=result['stderr']))
        boards.append(dict(port=port, metrics=metrics, python_versions={k: sorted(v) for k,v in versions.items()},
            python_builds_identical=versions['host_process'] == versions['cached_container'],
            image_digest=board['image_digest'], architecture=board['preflight']['architecture'],
            kernel=board['preflight']['kernel'], acquisitions=acquisitions))
    analysis = root/'analysis'
    analysis.mkdir(exist_ok=False)
    audit = dict(complete=True, activation_count=len(manifest['trials']),
        individual_predictions=len(manifest['trials'])*(1+RESIDENT_CALLS), paired_blocks_per_board=BLOCKS,
        image_acquisitions=len(PORTS)*PULLS, raw_sha256=raw_checksums,
        interpretation='Deployment-stack contrast, not pure isolation cost; all acquisition attempts reported',
        boundaries='SSH/CLI/model transfer included in launch timings; no image pull in cached activation')
    (analysis/'summary.json').write_text(json.dumps(boards, indent=2), encoding='utf-8')
    def cell(value):
        return r'%.2f $\pm$ %.2f' % (value['mean'], value['halfwidth95'])
    lines = [r'\begin{tabular}{llrrrr}', r'\toprule',
        r'Port & Deployment & Ready (ms) & First result (ms) & Resident (ms) & Compute (ms) \\', r'\midrule']
    for board in boards:
        for method, label in [('host_process', 'Host process'), ('cached_container', 'Cached container'),
                              ('paired_container_minus_host', 'Paired difference')]:
            vals = [cell(board['metrics'][k][method]) for k in
                    ['ready_ms', 'first_prediction_ms', 'resident_response_ms', 'resident_compute_ms']]
            lines.append(r'%d & %s & %s \\' % (board['port'], label, ' & '.join(vals)))
        lines.append(r'\addlinespace')
    lines += [r'\bottomrule', r'\end{tabular}']
    (analysis/'container_activation_table.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    # Completion sentinel last, after all statistics and tables have been written.
    (analysis/'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in audit.items() if k != 'raw_sha256'}), flush=True)


if __name__ == '__main__':
    main()
