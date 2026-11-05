"""Full low-duty calibration of the trained MLP on the identified live boards.

Twenty repetitions per batch size, persisted per response. Not a scheduling
experiment: these measurements calibrate the later physical controller.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import time
import numpy as np

from src.board_inference_client import BoardClient


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    provenance = dict(started_utc=datetime.now(timezone.utc).isoformat(),
                      source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (Path(__file__), Path(__file__).with_name('board_inference_client.py'),
                                   Path(__file__).with_name('board_inference_worker.py'),
                                   Path(__file__).with_name('portable_inference.py'))},
                      inventory_sha256=hashlib.sha256(args.inventory.read_bytes()).hexdigest(),
                      settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    (args.out/'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    rows = json.loads(args.inventory.read_text(encoding='utf-8'))['ports']
    rows = [r for r in rows if r['port'] not in (2, 15) and r.get('ssh_login') is True
            and (not args.ports or r['port'] in args.ports)]
    model_bytes = (args.model/'model.json').read_bytes()
    model = json.loads(model_bytes)
    validation = np.load(args.model/'validation_inputs.npz')
    features, expected = validation['features'], validation['expected']
    jobs = [(batch, repetition) for batch in (1, 4, 16, 64) for repetition in range(args.repetitions)]
    random.Random(9017).shuffle(jobs)
    key = os.environ['EDGE_SSH_KEY']

    def measure(row):
        report = dict(port=row['port'], ip=row['ip'], model_sha256=hashlib.sha256(model_bytes).hexdigest(),
                      started_utc=datetime.now(timezone.utc).isoformat(), requested_jobs=len(jobs))
        client = None
        succeeded = guarded = failed = 0
        try:
            client = BoardClient(row, key, model, args.temperature_limit)
            report['worker_ready'] = client.ready
            with (args.out/('port_%02d.jsonl' % row['port'])).open('x', encoding='utf-8') as log:
                for ordinal, (batch, repetition) in enumerate(jobs):
                    indices = (np.arange(batch)+repetition*7) % len(features)
                    result = client.infer(ordinal, features[indices])
                    result.update(repetition=repetition, port=row['port'],
                                  recorded_utc=datetime.now(timezone.utc).isoformat())
                    if result['status'] == 'ok':
                        actual = np.asarray(result['predictions'])
                        error = np.abs(actual-expected[indices])/np.maximum(np.abs(expected[indices]), 1.)
                        result['max_relative_error'] = float(error.max())
                        result['parity_ok'] = bool(error.max() < 1e-4)
                        if not result['parity_ok']:
                            result['status'] = 'parity_failed'
                            failed += 1
                        else:
                            succeeded += 1
                    elif result['status'] == 'temperature_guard':
                        guarded += 1
                    else:
                        failed += 1
                    log.write(json.dumps(result, allow_nan=False)+'\n')
                    log.flush()
                    if result['status'] != 'ok':
                        # Fail closed and preserve the partial run. No blind
                        # retry or reduced safety threshold to complete a table.
                        break
                    compute_s = result['compute_end']-result['compute_start']
                    time.sleep(min(30., max(.2, 2.*compute_s)))
        except Exception as exc:
            report['error'] = repr(exc)
            failed += 1
        finally:
            if client:
                report['cleanup'] = client.close()
        report.update(succeeded=succeeded, guarded=guarded, failed=failed,
                      complete=(succeeded == len(jobs)), ended_utc=datetime.now(timezone.utc).isoformat())
        (args.out/('port_%02d_summary.json' % row['port'])).write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps({k: report[k] for k in ('port', 'succeeded', 'guarded', 'failed', 'complete')}, ensure_ascii=False), flush=True)
        return report

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        reports = [f.result() for f in as_completed([pool.submit(measure, r) for r in rows])]
    summary = dict(protocol='20 repeats per batch 1/4/16/64, shared randomized order, CPU reference MLP',
                   repetitions=args.repetitions, boards=sorted(reports, key=lambda r: r['port']),
                   all_complete=all(r['complete'] for r in reports), temperature_limit_c=args.temperature_limit,
                   timing_scope='worker compute interval includes recorded safety checks; controller latency includes SSH transport',
                   claim='Inference calibration and output parity; not repeated scheduling or container cold-start evidence')
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(dict(boards=len(rows), complete=sum(r['complete'] for r in reports),
                         successful_jobs=sum(r['succeeded'] for r in reports))), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--inventory', type=Path, required=True)
    p.add_argument('--model', type=Path, default=Path('results/revision/physical/portable_model_20260905'))
    p.add_argument('--out', type=Path, default=Path('results/revision/physical/inference_calibration_20260905'))
    p.add_argument('--repetitions', type=int, default=20)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--ports', nargs='+', type=int)
    p.add_argument('--temperature-limit', type=float, default=65.)
    run(p.parse_args())
